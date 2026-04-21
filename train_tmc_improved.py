import argparse
import gc
import math
import os
import random
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

from AMDGT_original.data_preprocess import (
    data_processing,
    dgl_heterograph,
    get_data,
    k_fold,
)
from data_preprocess_improved import dgl_similarity_view_graphs
from metric import get_metric
from model.improved.tmc_rvg_model import TMC_AMDGT_RVG
from similarity_fusion_improved import collect_similarity_views, repair_similarity_views
from topology_features_improved import extract_topology_features


DATASET_PRESETS = {
    'B-dataset': {'neighbor': 3, 'gt_out_dim': 512, 'hgt_in_dim': 512, 'hgt_head_dim': 64, 'hgt_out_dim': 512},
    'C-dataset': {'neighbor': 5, 'gt_out_dim': 256, 'hgt_in_dim': 256, 'hgt_head_dim': 32, 'hgt_out_dim': 256},
    'F-dataset': {'neighbor': 5, 'gt_out_dim': 256, 'hgt_in_dim': 256, 'hgt_head_dim': 32, 'hgt_out_dim': 256},
}


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def update_ema(ema_state, model_state, decay):
    if ema_state is None:
        return {key: value.detach().cpu().clone() for key, value in model_state.items()}
    for key, value in model_state.items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach().cpu(), alpha=1.0 - decay)
        else:
            ema_state[key] = value.detach().cpu().clone()
    return ema_state


def build_scheduler(optimizer, args):
    warmup_epochs = max(1, min(args.lr_warmup_epochs, args.epochs))
    min_scale = min(args.min_lr / max(args.lr, 1e-8), 1.0)

    def lr_lambda(epoch_idx):
        step = epoch_idx + 1
        if step <= warmup_epochs:
            return step / warmup_epochs
        progress = (step - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def normalize_prior_matrix(matrix):
    matrix = matrix - matrix.min()
    return matrix / matrix.max().clamp_min(1e-6)


def prepare_similarity_tensor(matrix):
    similarity = torch.as_tensor(matrix, dtype=torch.float32)
    similarity = 0.5 * (similarity + similarity.T)
    similarity.fill_diagonal_(0.0)
    return similarity


def build_similarity_regularizer(graph, similarity_matrix, device):
    src, dst = graph.edges()
    src = src.to(device)
    dst = dst.to(device)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    weights = similarity_matrix[src, dst]
    weights = normalize_prior_matrix(weights)
    return src, dst, weights


def graph_smoothness_loss(embeddings, regularizer_data, max_edges=None):
    src, dst, weights = regularizer_data
    if src.numel() == 0:
        return embeddings.new_tensor(0.0)
    if max_edges is not None and src.numel() > max_edges:
        idx = torch.randperm(src.numel(), device=embeddings.device)[:max_edges]
        src = src[idx]
        dst = dst[idx]
        weights = weights[idx]
    diff = (embeddings[src] - embeddings[dst]).pow(2).mean(dim=-1)
    return (diff * (0.25 + 0.75 * weights)).mean()


def positive_training_edges(x_train, y_train):
    labels = np.asarray(y_train).reshape(-1).astype(int)
    return np.asarray(x_train)[labels == 1]


def positive_pair_topology_loss(drug_repr, disease_repr, positive_edges, max_pairs=None):
    if positive_edges.numel() == 0:
        return drug_repr.new_tensor(0.0)
    if max_pairs is not None and positive_edges.shape[0] > max_pairs:
        idx = torch.randperm(positive_edges.shape[0], device=positive_edges.device)[:max_pairs]
        positive_edges = positive_edges[idx]
    drug_nodes = drug_repr[positive_edges[:, 0]]
    disease_nodes = disease_repr[positive_edges[:, 1]]
    cosine = fn.cosine_similarity(drug_nodes, disease_nodes, dim=-1)
    return (1.0 - cosine).mean()


def build_multiview_collab_prior(data, train_drdi, args):
    drug_views = [prepare_similarity_tensor(matrix) for matrix in collect_similarity_views(data, 'drug')]
    disease_views = [prepare_similarity_tensor(matrix) for matrix in collect_similarity_views(data, 'disease')]

    disease_pos_counts = train_drdi.sum(dim=0, keepdim=True).clamp_min(1.0)
    drug_pos_counts = train_drdi.sum(dim=1, keepdim=True).clamp_min(1.0)

    support_views = []
    for drug_similarity in drug_views:
        support_views.append((drug_similarity @ train_drdi) / disease_pos_counts)
    for disease_similarity in disease_views:
        support_views.append((train_drdi @ disease_similarity) / drug_pos_counts)

    stacked = torch.stack(support_views, dim=0)
    mean_support = stacked.mean(dim=0)
    if stacked.shape[0] == 1:
        return normalize_prior_matrix(mean_support)

    agreement = 1.0 - normalize_prior_matrix(stacked.std(dim=0, unbiased=False))
    return normalize_prior_matrix(
        (1.0 - args.prior_view_agreement_weight) * mean_support + args.prior_view_agreement_weight * agreement
    )


def build_path_prior(data, train_positive_edges, args):
    drug_n = args.drug_number
    disease_n = args.disease_number
    protein_n = args.protein_number

    train_drdi = torch.zeros((drug_n, disease_n), dtype=torch.float32)
    if len(train_positive_edges) > 0:
        train_pairs = torch.as_tensor(train_positive_edges, dtype=torch.long)
        train_drdi[train_pairs[:, 0], train_pairs[:, 1]] = 1.0

    drpr = torch.zeros((drug_n, protein_n), dtype=torch.float32)
    if data['drpr'].size > 0:
        drpr_idx = torch.as_tensor(data['drpr'], dtype=torch.long)
        drpr[drpr_idx[:, 0], drpr_idx[:, 1]] = 1.0

    dipr = torch.zeros((protein_n, disease_n), dtype=torch.float32)
    if data['dipr'].size > 0:
        dipr_idx = torch.as_tensor(data['dipr'], dtype=torch.long)
        dipr[dipr_idx[:, 1], dipr_idx[:, 0]] = 1.0

    shared_paths = drpr @ dipr
    shared_norm = normalize_prior_matrix(shared_paths)

    drug_deg = drpr.sum(dim=1, keepdim=True)
    disease_deg = dipr.sum(dim=0, keepdim=True)
    degree_mix = torch.sqrt((drug_deg + 1.0) * (disease_deg + 1.0))
    degree_norm = normalize_prior_matrix(degree_mix)

    collab_norm = build_multiview_collab_prior(data, train_drdi, args)

    train_assoc_norm = normalize_prior_matrix(train_drdi) if train_drdi.max() > 0 else train_drdi
    indirect_prior = normalize_prior_matrix(0.50 * shared_norm + 0.35 * collab_norm + 0.15 * degree_norm)
    train_prior = normalize_prior_matrix(
        (1.0 - args.direct_train_prior_weight) * indirect_prior + args.direct_train_prior_weight * train_assoc_norm
    )
    return indirect_prior, train_prior


def gather_pair_values(pair_index, value_matrix, device):
    idx = pair_index.long().detach().cpu().clone()
    idx[:, 0] = idx[:, 0].clamp(0, value_matrix.shape[0] - 1)
    idx[:, 1] = idx[:, 1].clamp(0, value_matrix.shape[1] - 1)
    return value_matrix[idx[:, 0], idx[:, 1]].to(device).unsqueeze(-1)


def gather_pair_bias(pair_index, prior_matrix, device, scale=0.22):
    return scale * gather_pair_values(pair_index, prior_matrix, device)


def contrastive_weight_for_epoch(epoch, args):
    if epoch + 1 <= args.cl_warmup_epochs:
        return args.lambda_cl
    progress = (epoch + 1 - args.cl_warmup_epochs) / max(1, args.epochs - args.cl_warmup_epochs)
    decay = (1.0 - min(max(progress, 0.0), 1.0)) ** 2
    return args.lambda_cl * (args.cl_min_scale + (1.0 - args.cl_min_scale) * decay)


def pair_ranking_loss(logits, targets, margin, max_pairs):
    probs = fn.softmax(logits, dim=-1)[:, 1]
    pos_scores = probs[targets == 1]
    neg_scores = probs[targets == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return logits.new_tensor(0.0)

    sample_count = min(int(max_pairs), pos_scores.numel(), neg_scores.numel())
    pos_idx = torch.randperm(pos_scores.numel(), device=logits.device)[:sample_count]
    neg_idx = torch.randperm(neg_scores.numel(), device=logits.device)[:sample_count]
    return torch.relu(margin - pos_scores[pos_idx] + neg_scores[neg_idx]).mean()


def hard_negative_mining_loss(logits, targets, top_ratio=0.15, margin=0.12):
    probs = fn.softmax(logits, dim=-1)[:, 1]
    pos_scores = probs[targets == 1]
    neg_scores = probs[targets == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return logits.new_tensor(0.0)

    top_k = max(1, int(top_ratio * neg_scores.numel()))
    hard_neg, _ = torch.topk(neg_scores, k=min(top_k, neg_scores.numel()))
    pos_ref = pos_scores.mean()
    return torch.relu(margin + hard_neg - pos_ref).mean()


def aux_loss_weights(epoch, args):
    if epoch + 1 <= args.aux_warmup_epochs:
        return 0.0, 0.0
    progress = (epoch + 1 - args.aux_warmup_epochs) / max(1, args.epochs - args.aux_warmup_epochs)
    ramp = min(max(progress, 0.0), 1.0)
    ranking = args.ranking_weight * (0.20 + 0.80 * ramp)
    hard_neg = args.hard_negative_weight * (0.15 + 0.85 * ramp)
    return ranking, hard_neg


def structure_loss_weights(epoch, args):
    if epoch + 1 <= args.aux_warmup_epochs:
        return 0.0, 0.0
    progress = (epoch + 1 - args.aux_warmup_epochs) / max(1, args.epochs - args.aux_warmup_epochs)
    ramp = min(max(progress, 0.0), 1.0)
    topology = args.topology_reg_weight * (0.20 + 0.80 * ramp)
    positive = args.positive_pair_reg_weight * (0.15 + 0.85 * ramp)
    return topology, positive


def classification_phase(epoch, args):
    if epoch + 1 <= args.aux_warmup_epochs:
        return {
            'hard_negative': 0.0,
            'hard_positive': 0.0,
            'prior_hardness': 0.0,
            'label_smoothing': args.label_smoothing,
        }

    progress = (epoch + 1 - args.aux_warmup_epochs) / max(1, args.epochs - args.aux_warmup_epochs)
    ramp = min(max(progress, 0.0), 1.0)
    train_progress = (epoch + 1) / max(1, args.epochs)
    if train_progress < 0.75:
        label_smoothing = args.label_smoothing
    elif train_progress < 0.90:
        label_smoothing = max(args.label_smoothing * 0.5, 0.001)
    else:
        label_smoothing = 0.0

    return {
        'hard_negative': args.ce_hard_negative_weight * (0.15 + 0.85 * ramp),
        'hard_positive': args.ce_hard_positive_weight * (0.15 + 0.85 * ramp),
        'prior_hardness': args.prior_hardness_weight * (0.15 + 0.85 * ramp),
        'label_smoothing': label_smoothing,
    }


def build_sample_weights(
    logits,
    targets,
    hard_negative_weight=0.0,
    hard_positive_weight=0.0,
    pair_prior=None,
    prior_hardness_weight=0.0,
):
    probs = fn.softmax(logits.detach(), dim=-1)[:, 1]
    sample_weights = torch.ones_like(probs)
    negative_mask = targets == 0
    positive_mask = targets == 1

    if hard_negative_weight > 0:
        sample_weights[negative_mask] = sample_weights[negative_mask] + hard_negative_weight * probs[negative_mask]
    if hard_positive_weight > 0:
        sample_weights[positive_mask] = sample_weights[positive_mask] + hard_positive_weight * (1.0 - probs[positive_mask])
    if pair_prior is not None and prior_hardness_weight > 0:
        prior = pair_prior.detach().reshape(-1).clamp(0.0, 1.0)
        sample_weights[negative_mask] = sample_weights[negative_mask] + prior_hardness_weight * prior[negative_mask]
        sample_weights[positive_mask] = sample_weights[positive_mask] + prior_hardness_weight * (1.0 - prior[positive_mask])

    # Re-center the weights so we emphasize harder pairs without silently inflating the total loss scale.
    return sample_weights / sample_weights.mean().clamp_min(1e-6)


def weighted_cross_entropy_loss(logits, targets, class_weights, label_smoothing, sample_weights=None):
    loss = fn.cross_entropy(
        logits,
        targets,
        reduction='none',
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def focal_classification_loss(logits, targets, class_weights, gamma, sample_weights=None):
    base_ce = fn.cross_entropy(logits, targets, reduction='none')
    weighted_ce = fn.cross_entropy(logits, targets, reduction='none', weight=class_weights)
    pt = torch.exp(-base_ce)
    loss = ((1.0 - pt) ** gamma) * weighted_ce
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def focal_weight_for_epoch(epoch, args):
    if epoch + 1 <= args.focal_start_epoch:
        return 0.0
    progress = (epoch + 1 - args.focal_start_epoch) / max(1, args.epochs - args.focal_start_epoch)
    ramp = min(max(progress, 0.0), 1.0)
    return args.focal_weight * (0.15 + 0.85 * ramp)


def build_results_dataframe(fold_metrics, fold_ids=None):
    columns = ['Fold', 'Best_Epoch', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Mcc']
    if fold_ids is None:
        fold_ids = list(range(len(fold_metrics['AUC'])))

    results_df = pd.DataFrame(
        {
            'Fold': [f'Fold {i}' for i in fold_ids],
            'Best_Epoch': fold_metrics['Best_Epoch'],
            'AUC': fold_metrics['AUC'],
            'AUPR': fold_metrics['AUPR'],
            'Accuracy': fold_metrics['Accuracy'],
            'Precision': fold_metrics['Precision'],
            'Recall': fold_metrics['Recall'],
            'F1-score': fold_metrics['F1-score'],
            'Mcc': fold_metrics['Mcc'],
        }
    )
    metric_columns = columns[2:]
    mean_row = {'Fold': 'Mean', 'Best_Epoch': ''}
    std_row = {'Fold': 'Std', 'Best_Epoch': ''}
    for metric in metric_columns:
        values = results_df[metric].to_numpy(dtype=np.float64)
        mean_row[metric] = float(np.mean(values))
        std_row[metric] = float(np.std(values))
    return pd.concat([results_df, pd.DataFrame([mean_row, std_row], columns=columns)], ignore_index=True)


def apply_dataset_preset(args):
    preset = DATASET_PRESETS.get(args.dataset, {})
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TMC-AMDGT-RVG Training')
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--fold_indices', nargs='+', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_warmup_epochs', type=int, default=40)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--neighbor', type=int, default=None)
    parser.add_argument('--negative_rate', type=float, default=1.0)
    parser.add_argument('--dataset', default='C-dataset')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gt_layer', type=int, default=2)
    parser.add_argument('--gt_head', type=int, default=2)
    parser.add_argument('--gt_out_dim', type=int, default=None)
    parser.add_argument('--hgt_layer', type=int, default=2)
    parser.add_argument('--hgt_head', type=int, default=8)
    parser.add_argument('--hgt_in_dim', type=int, default=None)
    parser.add_argument('--hgt_head_dim', type=int, default=None)
    parser.add_argument('--hgt_out_dim', type=int, default=None)
    parser.add_argument('--tr_layer', type=int, default=2)
    parser.add_argument('--tr_head', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--result_root', default=None)
    parser.add_argument('--score_every', type=int, default=1)
    parser.add_argument('--save_checkpoints', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--disable_scheduler', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--lambda_cl', type=float, default=0.1)
    parser.add_argument('--cl_warmup_epochs', type=int, default=200)
    parser.add_argument('--cl_min_scale', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--topo_hidden', type=int, default=128)
    parser.add_argument('--gate_mode', choices=['scalar', 'vector'], default='vector')
    parser.add_argument('--gate_bias_init', type=float, default=-2.0)
    parser.add_argument('--pair_decoder', choices=['hybrid_ensemble', 'hybrid_mlp', 'elementwise'], default='hybrid_ensemble')
    parser.add_argument('--path_bias_scale', type=float, default=0.18)
    parser.add_argument('--direct_train_prior_weight', type=float, default=0.18)
    parser.add_argument('--similarity_fusion', choices=['nonzero_mean', 'mean', 'legacy'], default='nonzero_mean')
    parser.add_argument('--prior_view_agreement_weight', type=float, default=0.15)
    parser.add_argument('--eval_path_bias', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--aux_warmup_epochs', type=int, default=180)
    parser.add_argument('--ranking_weight', type=float, default=0.06)
    parser.add_argument('--ranking_margin', type=float, default=0.18)
    parser.add_argument('--ranking_samples', type=int, default=2048)
    parser.add_argument('--hard_negative_weight', type=float, default=0.04)
    parser.add_argument('--hard_negative_ratio', type=float, default=0.15)
    parser.add_argument('--hard_negative_margin', type=float, default=0.10)
    parser.add_argument('--ce_hard_negative_weight', type=float, default=0.25)
    parser.add_argument('--ce_hard_positive_weight', type=float, default=0.10)
    parser.add_argument('--prior_hardness_weight', type=float, default=0.12)
    parser.add_argument('--topology_reg_weight', type=float, default=0.006)
    parser.add_argument('--positive_pair_reg_weight', type=float, default=0.012)
    parser.add_argument('--reg_edge_samples', type=int, default=12000)
    parser.add_argument('--reg_positive_samples', type=int, default=2048)
    parser.add_argument('--focal_weight', type=float, default=0.05)
    parser.add_argument('--focal_gamma', type=float, default=1.4)
    parser.add_argument('--focal_start_epoch', type=int, default=220)
    parser.add_argument('--label_smoothing', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--log_best_only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--display_best_delta', type=float, default=0.001, help='minimum AUC gain required before printing a new BEST line')

    args = parser.parse_args()
    apply_dataset_preset(args)
    args.direct_train_prior_weight = min(max(args.direct_train_prior_weight, 0.0), 1.0)
    args.cl_min_scale = min(max(args.cl_min_scale, 0.0), 1.0)
    args.prior_view_agreement_weight = min(max(args.prior_view_agreement_weight, 0.0), 1.0)
    args.ce_hard_negative_weight = max(args.ce_hard_negative_weight, 0.0)
    args.ce_hard_positive_weight = max(args.ce_hard_positive_weight, 0.0)
    args.prior_hardness_weight = max(args.prior_hardness_weight, 0.0)
    args.topology_reg_weight = max(args.topology_reg_weight, 0.0)
    args.positive_pair_reg_weight = max(args.positive_pair_reg_weight, 0.0)
    args.topo_feat_dim = 7
    args.device = resolve_device(args.device)
    set_seed(args.random_seed)

    default_data_dir = Path('AMDGT_original') / 'data' / args.dataset
    default_result_dir = Path('Result') / 'tmc_improved' / args.dataset
    args.data_dir = str(Path(args.data_root) if args.data_root else default_data_dir) + os.sep
    args.result_dir = str(Path(args.result_root) if args.result_root else default_result_dir)
    os.makedirs(args.result_dir, exist_ok=True)

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = repair_similarity_views(data, mode=args.similarity_fusion)
    data = k_fold(data, args)

    if args.fold_indices is None:
        selected_folds = list(range(args.k_fold))
    else:
        selected_folds = list(dict.fromkeys(args.fold_indices))
        invalid = [i for i in selected_folds if i < 0 or i >= args.k_fold]
        if invalid:
            raise ValueError(f'Invalid fold indices {invalid}; valid range is 0..{args.k_fold - 1}')

    drug_view_graphs, disease_view_graphs, data = dgl_similarity_view_graphs(data, args)
    drug_view_graphs = {name: graph.to(args.device) for name, graph in drug_view_graphs.items()}
    disease_view_graphs = {name: graph.to(args.device) for name, graph in disease_view_graphs.items()}
    drug_similarity_matrix = prepare_similarity_tensor(data['drs']).to(args.device)
    disease_similarity_matrix = prepare_similarity_tensor(data['dis']).to(args.device)
    drug_similarity_reg = build_similarity_regularizer(drug_view_graphs['consensus'], drug_similarity_matrix, args.device)
    disease_similarity_reg = build_similarity_regularizer(
        disease_view_graphs['consensus'], disease_similarity_matrix, args.device
    )

    drug_topo_feat, disease_topo_feat = extract_topology_features(data, args)
    drug_topo_feat = drug_topo_feat.to(args.device)
    disease_topo_feat = disease_topo_feat.to(args.device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(args.device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(args.device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(args.device)

    print('--- Starting TMC-AMDGT-RVG Pipeline ---')
    print(f'Dataset: {args.dataset} | Device: {args.device} | Result dir: {args.result_dir}')
    print(f'Running folds: {selected_folds}')
    print('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')

    aucs, auprs, accs, precs, recs, f1s, mccs, epochs = [], [], [], [], [], [], [], []
    global_start = timeit.default_timer()

    for fold_idx in selected_folds:
        print(f'\n--- Fold: {fold_idx} ---')
        model = TMC_AMDGT_RVG(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None if args.disable_scheduler else build_scheduler(optimizer, args)

        x_train = torch.LongTensor(data['X_train'][fold_idx]).to(args.device)
        y_train = torch.LongTensor(data['Y_train'][fold_idx]).to(args.device).flatten()
        x_test = torch.LongTensor(data['X_test'][fold_idx]).to(args.device)
        y_test = data['Y_test'][fold_idx].flatten()

        n_pos = torch.sum(y_train).item()
        n_neg = y_train.numel() - n_pos
        class_weights = torch.tensor([1.0, max(n_neg / max(n_pos, 1.0), 1.0)], device=args.device)

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][fold_idx], args)
        drdipr_graph = drdipr_graph.to(args.device)
        train_positive_edges = positive_training_edges(data['X_train'][fold_idx], data['Y_train'][fold_idx])
        train_positive_edges_tensor = torch.as_tensor(train_positive_edges, dtype=torch.long, device=args.device)
        eval_prior, train_prior = build_path_prior(data, train_positive_edges, args)
        eval_prior = eval_prior.to(args.device)
        train_prior = train_prior.to(args.device)

        best_metrics = None
        best_auc = -1.0
        displayed_best_auc = -1.0
        no_improve_epochs = 0
        ema_state_dict = None

        for epoch in range(args.epochs):
            model.train()
            cl_weight = contrastive_weight_for_epoch(epoch, args)
            ranking_weight, hard_neg_weight = aux_loss_weights(epoch, args)
            topology_reg_weight, positive_reg_weight = structure_loss_weights(epoch, args)
            focal_weight = focal_weight_for_epoch(epoch, args)
            classification_cfg = classification_phase(epoch, args)
            train_pair_prior = gather_pair_values(x_train, train_prior, args.device)
            train_edge_stats = {
                'pair_bias': args.path_bias_scale * train_pair_prior
            }
            _, train_score, aux_losses = model(
                drug_view_graphs,
                disease_view_graphs,
                drdipr_graph,
                drug_feature,
                disease_feature,
                protein_feature,
                drug_topo_feat,
                disease_topo_feat,
                x_train,
                edge_stats=train_edge_stats,
                return_aux=True,
            )
            cl_loss = aux_losses['contrastive']
            sample_weights = build_sample_weights(
                train_score,
                y_train,
                hard_negative_weight=classification_cfg['hard_negative'],
                hard_positive_weight=classification_cfg['hard_positive'],
                pair_prior=train_pair_prior,
                prior_hardness_weight=classification_cfg['prior_hardness'],
            )
            ce_loss = weighted_cross_entropy_loss(
                train_score,
                y_train,
                class_weights,
                classification_cfg['label_smoothing'],
                sample_weights=sample_weights,
            )
            ranking_loss = pair_ranking_loss(train_score, y_train, args.ranking_margin, args.ranking_samples)
            hard_neg_loss = hard_negative_mining_loss(
                train_score, y_train, top_ratio=args.hard_negative_ratio, margin=args.hard_negative_margin
            )
            topology_loss = graph_smoothness_loss(
                aux_losses['drug_repr'], drug_similarity_reg, max_edges=args.reg_edge_samples
            )
            topology_loss = topology_loss + graph_smoothness_loss(
                aux_losses['disease_repr'], disease_similarity_reg, max_edges=args.reg_edge_samples
            )
            positive_reg_loss = positive_pair_topology_loss(
                aux_losses['drug_repr'],
                aux_losses['disease_repr'],
                train_positive_edges_tensor,
                max_pairs=args.reg_positive_samples,
            )
            focal_loss = focal_classification_loss(
                train_score,
                y_train,
                class_weights,
                args.focal_gamma,
                sample_weights=sample_weights,
            )
            train_loss = (
                ce_loss
                + cl_weight * cl_loss
                + ranking_weight * ranking_loss
                + hard_neg_weight * hard_neg_loss
                + topology_reg_weight * topology_loss
                + positive_reg_weight * positive_reg_loss
                + focal_weight * focal_loss
            )

            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            ema_state_dict = update_ema(ema_state_dict, model.state_dict(), args.ema_decay)

            if (epoch + 1) % max(1, args.score_every) != 0:
                continue

            backup_state = None
            if ema_state_dict is not None:
                backup_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(ema_state_dict, strict=False)
            model.eval()
            eval_edge_stats = None
            if args.eval_path_bias:
                eval_edge_stats = {
                    'pair_bias': gather_pair_bias(x_test, eval_prior, args.device, scale=args.path_bias_scale)
                }
            with torch.no_grad():
                _, test_score, _ = model(
                    drug_view_graphs,
                    disease_view_graphs,
                    drdipr_graph,
                    drug_feature,
                    disease_feature,
                    protein_feature,
                    drug_topo_feat,
                    disease_topo_feat,
                    x_test,
                    edge_stats=eval_edge_stats,
                )
            if backup_state is not None:
                model.load_state_dict(backup_state, strict=False)

            test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
            test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
            auc, aupr, accuracy, precision, recall, f1, mcc = get_metric(y_test, test_pred, test_prob)

            if (auc > best_auc + 1e-6) or (
                abs(auc - best_auc) <= 1e-6 and best_metrics is not None and aupr > best_metrics[1] + 1e-6
            ) or (
                abs(auc - best_auc) <= 1e-6 and best_metrics is None
            ):
                best_auc = auc
                no_improve_epochs = 0
                best_metrics = (auc, aupr, accuracy, precision, recall, f1, mcc, epoch + 1)
                if args.save_checkpoints:
                    state_to_save = ema_state_dict if ema_state_dict is not None else model.state_dict()
                    torch.save(state_to_save, os.path.join(args.result_dir, f'best_model_fold_{fold_idx}.pth'))
            else:
                no_improve_epochs += max(1, args.score_every)

            elapsed = timeit.default_timer() - global_start
            best_mark = ' [BEST]' if abs(auc - best_auc) < 1e-12 else ''
            should_print_best = False
            if best_mark:
                if displayed_best_auc < 0:
                    should_print_best = True
                elif auc >= displayed_best_auc + args.display_best_delta - 1e-12:
                    should_print_best = True
                if should_print_best:
                    displayed_best_auc = auc
            if (not args.log_best_only) or should_print_best:
                print(
                    f'Epoch {epoch + 1:4d} | {elapsed:7.2f}s | '
                    f'AUC {auc:.5f} | AUPR {aupr:.5f} | ACC {accuracy:.5f} | '
                    f'P {precision:.5f} | R {recall:.5f} | F1 {f1:.5f} | MCC {mcc:.5f}'
                    f'{best_mark} | NO_IMPROVE {no_improve_epochs}'
                )

        if best_metrics is None:
            raise RuntimeError(f'No evaluation executed for fold {fold_idx}; check score_every/epochs.')

        aucs.append(best_metrics[0])
        auprs.append(best_metrics[1])
        accs.append(best_metrics[2])
        precs.append(best_metrics[3])
        recs.append(best_metrics[4])
        f1s.append(best_metrics[5])
        mccs.append(best_metrics[6])
        epochs.append(best_metrics[7])
        print(f'Fold {fold_idx} summary -> best AUC {best_metrics[0]:.5f} at epoch {best_metrics[7]}')

        del model, optimizer, scheduler, drdipr_graph
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_df = build_results_dataframe(
        {
            'Best_Epoch': epochs,
            'AUC': aucs,
            'AUPR': auprs,
            'Accuracy': accs,
            'Precision': precs,
            'Recall': recs,
            'F1-score': f1s,
            'Mcc': mccs,
        },
        fold_ids=selected_folds,
    )

    print('\n' + '=' * 30 + '\nFINAL RESULTS SUMMARY (TMC-AMDGT-RVG)\n' + '=' * 30)
    print(final_df.iloc[-2:])

    if len(selected_folds) == args.k_fold and selected_folds == list(range(args.k_fold)):
        csv_name = '10_fold_results_improved.csv'
    else:
        csv_name = f"selected_fold_results_improved_{'_'.join(map(str, selected_folds))}.csv"
    csv_path = os.path.join(args.result_dir, csv_name)
    final_df.to_csv(csv_path, index=False)
    print(f'\nSaved TMC results to: {csv_path}')
