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

from data_preprocess_improved import (
    dgl_heterograph,
    dgl_similarity_view_graphs,
    data_processing,
    get_data,
    k_fold,
)
from metric import get_metric
from model.improved.improved_model import AMNTDDA


REQUIRED_DATA_FILES = [
    'DrugFingerprint.csv',
    'DrugGIP.csv',
    'DiseasePS.csv',
    'DiseaseGIP.csv',
    'DrugDiseaseAssociationNumber.csv',
    'DrugProteinAssociationNumber.csv',
    'ProteinDiseaseAssociationNumber.csv',
    'Drug_mol2vec.csv',
    'DiseaseFeature.csv',
    'Protein_ESM.csv',
]


def resolve_device(device_name):
    if device_name == 'auto':
        device_name = os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_data_dir(data_dir):
    missing = [name for name in REQUIRED_DATA_FILES if not os.path.exists(os.path.join(data_dir, name))]
    if missing:
        joined = ', '.join(missing)
        raise FileNotFoundError(f'Missing dataset files in {data_dir}: {joined}')


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


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


def update_ema(ema_state, model_state, decay):
    if ema_state is None:
        return {k: v.detach().cpu().clone() for k, v in model_state.items()}
    for key, value in model_state.items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach().cpu(), alpha=1.0 - decay)
        else:
            ema_state[key] = value.detach().cpu().clone()
    return ema_state


def normalize_prior_matrix(matrix):
    matrix = matrix - matrix.min()
    return matrix / matrix.max().clamp_min(1e-6)


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


def attention_sparsity_loss(aux_losses):
    penalty = 0.0
    eps = 1e-8
    for key in ('drug_view_weights', 'disease_view_weights', 'drug_token_weights', 'disease_token_weights'):
        weights = aux_losses[key].clamp_min(eps)
        penalty = penalty + (-(weights * torch.log(weights)).sum(dim=-1).mean())
    return penalty / 4.0


def modality_gate_regularization(aux_losses):
    penalty = 0.0
    for key in ('drug_view_gates', 'disease_view_gates', 'drug_token_gates', 'disease_token_gates'):
        penalty = penalty + aux_losses[key].mean()
    return penalty / 4.0


def weighted_classification_loss(logits, targets, class_weights, focal_criterion, label_smoothing, hard_negative_weight, use_focal):
    probs = fn.softmax(logits.detach(), dim=-1)[:, 1]
    sample_weights = torch.ones_like(probs)
    negative_mask = targets == 0
    sample_weights[negative_mask] = 1.0 + hard_negative_weight * probs[negative_mask]

    ce_loss = fn.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction='none',
        label_smoothing=label_smoothing,
    )
    ce_loss = (ce_loss * sample_weights).mean()
    if not use_focal:
        return ce_loss

    focal_loss = focal_criterion(logits, targets)
    focal_loss = (focal_loss * sample_weights).mean()
    return 0.5 * ce_loss + 0.5 * focal_loss


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


def positive_training_edges(x_train, y_train):
    labels = np.asarray(y_train).reshape(-1).astype(int)
    return np.asarray(x_train)[labels == 1]


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
        # CSV stores (disease, protein); convert to (protein, disease)
        dipr[dipr_idx[:, 1], dipr_idx[:, 0]] = 1.0

    shared_paths = drpr @ dipr
    shared_norm = normalize_prior_matrix(shared_paths)

    drug_deg = drpr.sum(dim=1, keepdim=True)
    disease_deg = dipr.sum(dim=0, keepdim=True)
    degree_mix = torch.sqrt((drug_deg + 1.0) * (disease_deg + 1.0))
    degree_norm = normalize_prior_matrix(degree_mix)

    # Collaborative prior: a drug gets support from drugs similar to those
    # already linked with the target disease, and vice versa for diseases.
    drug_similarity = torch.as_tensor(data['drs'], dtype=torch.float32)
    disease_similarity = torch.as_tensor(data['dis'], dtype=torch.float32)
    drug_similarity.fill_diagonal_(0.0)
    disease_similarity.fill_diagonal_(0.0)

    disease_pos_counts = train_drdi.sum(dim=0, keepdim=True).clamp_min(1.0)
    drug_pos_counts = train_drdi.sum(dim=1, keepdim=True).clamp_min(1.0)
    drug_support = (drug_similarity @ train_drdi) / disease_pos_counts
    disease_support = (train_drdi @ disease_similarity) / drug_pos_counts
    collab_norm = normalize_prior_matrix(0.5 * (drug_support + disease_support))

    # Do not inject the direct train association itself; that hurts
    # generalization because test positives are unseen by design.
    combined_prior = 0.45 * shared_norm + 0.40 * collab_norm + 0.15 * degree_norm
    return normalize_prior_matrix(combined_prior)


def gather_pair_bias(pair_index, prior_matrix, device, scale=0.22):
    idx = pair_index.long().detach().cpu()
    idx = idx.clone()
    idx[:, 0] = idx[:, 0].clamp(0, prior_matrix.shape[0] - 1)
    idx[:, 1] = idx[:, 1].clamp(0, prior_matrix.shape[1] - 1)
    bias = prior_matrix[idx[:, 0], idx[:, 1]].to(device)
    return scale * bias.unsqueeze(-1)


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


def phase_weights(epoch, args):
    progress = (epoch + 1) / max(1, args.epochs)
    # Keep the first phase classification-only to prevent early loss spikes.
    if epoch + 1 <= args.warmup_epochs:
        return {
            'ranking': 0.0,
            'contrastive': 0.0,
            'hard_neg': 0.0,
            'topology': 0.0,
            'positive_reg': 0.0,
            'attention_sparsity': 0.0,
            'modality_gate': 0.0,
            'hard_neg_scale': 1.0,
            'label_smoothing': args.label_smoothing,
        }
    ramp = min(1.0, (progress - (args.warmup_epochs / max(1, args.epochs))) / 0.35)
    ranking = args.ranking_weight * (0.15 + 0.85 * ramp)
    contrastive = args.contrastive_weight * max(0.0, (1.0 - ramp) ** 2)
    hard_neg = 0.02 + 0.10 * ramp
    topology = 0.20 + 0.80 * ramp
    positive_reg = 0.15 + 0.85 * ramp
    attention_sparsity = 0.10 + 0.90 * ramp
    modality_gate = 0.05 + 0.95 * ramp
    hard_neg_scale = 1.0 + 0.18 * ramp
    if progress < 0.75:
        label_smoothing = args.label_smoothing
    elif progress < 0.9:
        label_smoothing = max(args.label_smoothing * 0.5, 0.001)
    else:
        label_smoothing = 0.0
    return {
        'ranking': ranking,
        'contrastive': contrastive,
        'hard_neg': hard_neg,
        'topology': topology,
        'positive_reg': positive_reg,
        'attention_sparsity': attention_sparsity,
        'modality_gate': modality_gate,
        'hard_neg_scale': hard_neg_scale,
        'label_smoothing': label_smoothing,
    }


def build_results_dataframe(fold_metrics, fold_ids=None):
    columns = ['Fold', 'Best_Epoch', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Mcc']
    metric_columns = columns[2:]
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

    summary_rows = []
    for label, reducer in (('Mean', np.mean), ('Std', np.std)):
        row = {'Fold': label, 'Best_Epoch': ''}
        for metric in metric_columns:
            row[metric] = float(reducer(results_df[metric].to_numpy(dtype=np.float64)))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows, columns=columns)
    return pd.concat([results_df, summary_df], ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--fold_indices', nargs='+', type=int, default=None, help='optional subset of fold indices to run, e.g. --fold_indices 1 2 4 8')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate for cosine schedule')
    parser.add_argument('--lr_warmup_epochs', type=int, default=40, help='learning-rate warmup epochs')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=10, help='k for similarity knn graphs')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative sampling rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='training device')
    parser.add_argument('--data_root', default=None, help='dataset directory; defaults to AMDGT_original/data/<dataset>')
    parser.add_argument('--result_root', default=None, help='output directory; defaults to Result/improved/<dataset>')
    parser.add_argument('--save_checkpoints', action=argparse.BooleanOptionalAction, default=False, help='save model checkpoints to result_root')
    parser.add_argument('--warmup_epochs', default=150, type=int, help='epochs to train before enabling focal/ranking-heavy fine-tune')
    parser.add_argument('--eval_start_epoch', default=50, type=int, help='minimum epochs before evaluation begins')
    parser.add_argument('--score_every', default=10, type=int, help='evaluate every N epochs after eval start')
    parser.add_argument('--log_every', default=25, type=int, help='print training loss every N epochs')
    parser.add_argument('--focal_gamma', default=1.2, type=float, help='focal loss gamma during early training')
    parser.add_argument('--focal_gamma_warm', default=2.0, type=float, help='focal loss gamma during late training')
    parser.add_argument('--contrastive_weight', default=0.03, type=float, help='weight of node-level cross-view contrastive loss')
    parser.add_argument('--contrastive_temperature', default=0.2, type=float, help='temperature for contrastive loss')
    parser.add_argument('--ranking_weight', default=0.12, type=float, help='weight of pairwise ranking loss')
    parser.add_argument('--ranking_margin', default=0.2, type=float, help='margin used in ranking loss')
    parser.add_argument('--ranking_samples', default=2048, type=int, help='maximum positive-negative pairs used in ranking loss')
    parser.add_argument('--hard_negative_weight', default=2.0, type=float, help='reweight difficult negatives based on current positive score')
    parser.add_argument('--path_bias_scale', default=0.30, type=float, help='strength of the indirect topology prior used at train and eval time')
    parser.add_argument('--topology_reg_weight', default=0.03, type=float, help='smoothness regularization strength on similarity graphs')
    parser.add_argument('--positive_pair_reg_weight', default=0.015, type=float, help='topological regularization strength on known positive drug-disease pairs')
    parser.add_argument('--attention_sparsity_weight', default=0.004, type=float, help='entropy penalty that encourages selective modality usage')
    parser.add_argument('--modality_gate_weight', default=0.003, type=float, help='L1-style penalty that encourages skipping noisy modalities through selective gates')
    parser.add_argument('--reg_edge_samples', default=12000, type=int, help='maximum similarity edges sampled for topology regularization each step')
    parser.add_argument('--reg_positive_samples', default=2048, type=int, help='maximum positive drug-disease pairs sampled for topology regularization each step')
    parser.add_argument('--label_smoothing', default=0.01, type=float, help='label smoothing for cross entropy')
    parser.add_argument('--patience', default=150, type=int, help='early stopping patience in epochs without AUC improvement')
    parser.add_argument('--target_auc', default=0.96, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--target_auc_warmup', default=400, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--target_auc_patience', default=4, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--plateau_patience', default=3, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--plateau_factor', default=0.5, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--ema_decay', default=0.995, type=float, help='EMA decay for stable validation snapshots')

    parser.add_argument('--hgt_in_dim', default=96, type=int, help='HGT input dimension')
    parser.add_argument('--hgt_layer', default=3, type=int, help='HGT layers')
    parser.add_argument('--hgt_head', default=8, type=int, help='HGT heads')
    parser.add_argument('--gt_layer', default=2, type=int, help='GT layers')
    parser.add_argument('--gt_head', default=2, type=int, help='GT heads')
    parser.add_argument('--gt_out_dim', default=160, type=int, help='GT output dimension')
    parser.add_argument('--tr_layer', default=2, type=int, help='Transformer layers')
    parser.add_argument('--tr_head', default=4, type=int, help='Transformer heads')
    parser.add_argument('--use_relation_attention', action=argparse.BooleanOptionalAction, default=True, help='Use relation-aware attention in HGT')
    parser.add_argument('--use_metapath', action=argparse.BooleanOptionalAction, default=True, help='Use metapath aggregation')
    parser.add_argument('--use_global_hgt', action=argparse.BooleanOptionalAction, default=True, help='Use global context in HGT')
    parser.add_argument('--use_topological', action=argparse.BooleanOptionalAction, default=True, help='Use topological metapath projection')

    args = parser.parse_args()
    device = resolve_device(args.device)
    os.environ['AMDGT_DEVICE'] = device.type
    set_random_seed(args.random_seed)

    default_data_dir = Path('AMDGT_original') / 'data' / args.dataset
    default_result_dir = Path('Result') / 'improved' / args.dataset
    args.data_dir = str(Path(args.data_root) if args.data_root else default_data_dir)
    args.result_dir = str(Path(args.result_root) if args.result_root else default_result_dir)
    validate_data_dir(args.data_dir)
    os.makedirs(args.result_dir, exist_ok=True)

    print('--- Starting Final Improved Pipeline ---')
    print(f'Dataset: {args.dataset} | LR: {args.lr} | Dim: {args.gt_out_dim} | Neighbor: {args.neighbor}')
    print(f'Device: {device} | Data dir: {args.data_dir} | Result dir: {args.result_dir}')
    print(f'Save checkpoints: {args.save_checkpoints}')
    print(f"Early stopping is enabled: stop after {args.patience} epochs without AUC improvement.")

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    if args.fold_indices is None:
        selected_folds = list(range(args.k_fold))
    else:
        selected_folds = list(dict.fromkeys(args.fold_indices))
        invalid = [fold for fold in selected_folds if fold < 0 or fold >= args.k_fold]
        if invalid:
            raise ValueError(f'Invalid fold indices {invalid}; valid range is 0 to {args.k_fold - 1}')

    drug_view_graphs, disease_view_graphs, data = dgl_similarity_view_graphs(data, args)
    drug_view_graphs = {name: graph.to(device) for name, graph in drug_view_graphs.items()}
    disease_view_graphs = {name: graph.to(device) for name, graph in disease_view_graphs.items()}
    drug_similarity_matrix = torch.FloatTensor(data['drs']).to(device)
    disease_similarity_matrix = torch.FloatTensor(data['dis']).to(device)
    drug_similarity_reg = build_similarity_regularizer(drug_view_graphs['consensus'], drug_similarity_matrix, device)
    disease_similarity_reg = build_similarity_regularizer(disease_view_graphs['consensus'], disease_similarity_matrix, device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    metric_header = 'Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc'
    AUCs, AUPRs, Accs, Precs, Recs, F1s, MCCs, Epochs = [], [], [], [], [], [], [], []

    print(f'Running folds: {selected_folds}')

    for i in selected_folds:
        print(f'\n--- Fold: {i} ---')
        print(metric_header)

        model = AMNTDDA(args).to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = build_scheduler(optimizer, args)

        best_auc = -1.0
        best_metrics = None
        best_state_dict = None
        ema_state_dict = None
        no_improve_epochs = 0

        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device).flatten()
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        n_pos = torch.sum(Y_train).item()
        n_neg = Y_train.numel() - n_pos
        class_weights = torch.tensor([1.0, max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
        focal_criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, reduction='none')
        warm_focal_criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma_warm, reduction='none')

        train_positive_edges = positive_training_edges(data['X_train'][i], data['Y_train'][i])
        train_positive_edges_tensor = torch.LongTensor(train_positive_edges).to(device)
        drdipr_graph, data, edge_stats = dgl_heterograph(data, train_positive_edges, args)
        drdipr_graph = drdipr_graph.to(device)
        edge_stats = {k: v.to(device) for k, v in edge_stats.items()}
        path_prior = build_path_prior(data, train_positive_edges, args).to(device)

        start = timeit.default_timer()

        for epoch in range(args.epochs):
            model.train()
            phase = phase_weights(epoch, args)
            train_edge_stats = {
                'pair_bias': gather_pair_bias(X_train, path_prior, device, scale=args.path_bias_scale)
            }
            _, train_score, aux_losses = model(
                drug_view_graphs,
                disease_view_graphs,
                drdipr_graph,
                drug_feature,
                disease_feature,
                protein_feature,
                X_train,
                edge_stats=train_edge_stats,
                return_aux=True,
            )

            use_focal = (epoch + 1) > args.warmup_epochs
            focal_objective = warm_focal_criterion if use_focal else focal_criterion
            classification_loss = weighted_classification_loss(
                train_score,
                Y_train,
                class_weights,
                focal_objective,
                phase['label_smoothing'],
                args.hard_negative_weight * phase['hard_neg_scale'],
                use_focal,
            )
            ranking_loss = pair_ranking_loss(train_score, Y_train, args.ranking_margin, args.ranking_samples)
            hard_neg_loss = hard_negative_mining_loss(train_score, Y_train)
            contrastive_loss = aux_losses['contrastive']
            topology_loss = graph_smoothness_loss(aux_losses['drug_repr'], drug_similarity_reg, max_edges=args.reg_edge_samples)
            topology_loss = topology_loss + graph_smoothness_loss(aux_losses['disease_repr'], disease_similarity_reg, max_edges=args.reg_edge_samples)
            positive_reg = positive_pair_topology_loss(
                aux_losses['drug_repr'],
                aux_losses['disease_repr'],
                train_positive_edges_tensor,
                max_pairs=args.reg_positive_samples,
            )
            sparsity_loss = attention_sparsity_loss(aux_losses)
            gate_loss = modality_gate_regularization(aux_losses)
            train_loss = (
                classification_loss
                + phase['ranking'] * ranking_loss
                + phase['contrastive'] * contrastive_loss
                + phase['hard_neg'] * hard_neg_loss
                + (args.topology_reg_weight * phase['topology']) * topology_loss
                + (args.positive_pair_reg_weight * phase['positive_reg']) * positive_reg
                + (args.attention_sparsity_weight * phase['attention_sparsity']) * sparsity_loss
                + (args.modality_gate_weight * phase['modality_gate']) * gate_loss
            )

            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema_state_dict = update_ema(ema_state_dict, model.state_dict(), args.ema_decay)
            scheduler.step()

            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                elapsed = timeit.default_timer() - start
                print(
                    f'Epoch {epoch + 1:4d} | {elapsed:7.2f}s | loss {train_loss.item():.5f} | '
                    f'cls {classification_loss.item():.5f} | rank {ranking_loss.item():.5f} | '
                    f'ctr {contrastive_loss.item():.5f} | topo {topology_loss.item():.5f} | '
                    f'sparse {sparsity_loss.item():.5f} | gate {gate_loss.item():.5f} | lr {scheduler.get_last_lr()[0]:.6e}'
                )

            should_score = (epoch + 1) >= args.eval_start_epoch and ((epoch + 1 - args.eval_start_epoch) % max(1, args.score_every) == 0)
            if should_score:
                test_edge_stats = {
                    'pair_bias': gather_pair_bias(X_test, path_prior, device, scale=args.path_bias_scale)
                }
                eval_state = ema_state_dict if ema_state_dict is not None else None
                backup_state = None
                if eval_state is not None:
                    backup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(eval_state, strict=False)
                model.eval()
                with torch.no_grad():
                    _, test_score = model(
                        drug_view_graphs,
                        disease_view_graphs,
                        drdipr_graph,
                        drug_feature,
                        disease_feature,
                        protein_feature,
                        X_test,
                        edge_stats=test_edge_stats,
                    )
                if backup_state is not None:
                    model.load_state_dict(backup_state, strict=False)

                test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
                test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
                AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_pred, test_prob)
                stable_score = AUC + 0.10 * AUPR + 0.03 * f1

                if AUC > best_auc + 1e-6:
                    best_auc = AUC
                    best_metrics = (AUC, AUPR, accuracy, precision, recall, f1, mcc, epoch + 1)
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve_epochs = 0
                    if args.save_checkpoints:
                        torch.save(model.state_dict(), os.path.join(args.result_dir, f'best_model_fold_{i}.pth'))
                else:
                    no_improve_epochs += max(1, args.score_every)

                time_now = timeit.default_timer() - start
                best_mark = ' [BEST]' if abs(AUC - best_auc) < 1e-12 else ''
                print(
                    f'Epoch {epoch+1:4d} | {time_now:7.2f}s | '
                    f'AUC {AUC:.5f} | AUPR {AUPR:.5f} | ACC {accuracy:.5f} | '
                    f'P {precision:.5f} | R {recall:.5f} | F1 {f1:.5f} | MCC {mcc:.5f} | '
                    f'STABLE {stable_score:.5f}{best_mark} | NO_IMPROVE {no_improve_epochs}'
                )
                if no_improve_epochs >= args.patience:
                    print(f'Early stopping at epoch {epoch+1} after {no_improve_epochs} epochs without AUC improvement.')
                    break

        if best_metrics is None:
            test_edge_stats = {
                'pair_bias': gather_pair_bias(X_test, path_prior, device, scale=args.path_bias_scale)
            }
            eval_state = ema_state_dict if ema_state_dict is not None else None
            backup_state = None
            if eval_state is not None:
                backup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(eval_state, strict=False)
            model.eval()
            with torch.no_grad():
                _, test_score = model(
                    drug_view_graphs,
                    disease_view_graphs,
                    drdipr_graph,
                    drug_feature,
                    disease_feature,
                    protein_feature,
                    X_test,
                    edge_stats=test_edge_stats,
                )
            if backup_state is not None:
                model.load_state_dict(backup_state, strict=False)
            test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
            test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
            metrics = get_metric(Y_test, test_pred, test_prob)
            best_metrics = (*metrics, args.epochs)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        AUCs.append(best_metrics[0])
        AUPRs.append(best_metrics[1])
        Accs.append(best_metrics[2])
        Precs.append(best_metrics[3])
        Recs.append(best_metrics[4])
        F1s.append(best_metrics[5])
        MCCs.append(best_metrics[6])
        Epochs.append(best_metrics[7])
        if best_state_dict is not None and args.save_checkpoints:
            torch.save(best_state_dict, os.path.join(args.result_dir, f'best_model_fold_{i}_cpu.pth'))
            torch.save(ema_state_dict if ema_state_dict is not None else best_state_dict, os.path.join(args.result_dir, f'ema_model_fold_{i}.pth'))
        print(f'Fold {i} summary -> best AUC {best_metrics[0]:.5f} at epoch {best_metrics[7]}')

        del model, optimizer, scheduler, drdipr_graph
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_df = build_results_dataframe(
        {
            'Best_Epoch': Epochs,
            'AUC': AUCs,
            'AUPR': AUPRs,
            'Accuracy': Accs,
            'Precision': Precs,
            'Recall': Recs,
            'F1-score': F1s,
            'Mcc': MCCs,
        },
        fold_ids=selected_folds,
    )

    print('\n' + '=' * 30 + '\nFINAL RESULTS SUMMARY (IMPROVED PIPELINE)\n' + '=' * 30)
    print(final_df.iloc[-2:])

    if len(selected_folds) == args.k_fold and selected_folds == list(range(args.k_fold)):
        csv_name = '10_fold_results_improved.csv'
    else:
        fold_tag = '_'.join(str(fold) for fold in selected_folds)
        csv_name = f'selected_fold_results_improved_{fold_tag}.csv'
        print('Subset fold run detected: Mean/Std rows are computed only over the selected folds above.')
    csv_path = os.path.join(args.result_dir, csv_name)
    final_df.to_csv(csv_path, index=False)
    print(f'\nSaved improved results to: {csv_path}')
