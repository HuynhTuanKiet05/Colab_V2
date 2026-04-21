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
    dgl_similarity_graph,
    get_data,
    k_fold,
)
from metric import get_metric
from model.improved.tmc_rvg_model import TMC_AMDGT_RVG
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
        dipr[dipr_idx[:, 1], dipr_idx[:, 0]] = 1.0

    shared_paths = drpr @ dipr
    shared_norm = normalize_prior_matrix(shared_paths)

    drug_deg = drpr.sum(dim=1, keepdim=True)
    disease_deg = dipr.sum(dim=0, keepdim=True)
    degree_mix = torch.sqrt((drug_deg + 1.0) * (disease_deg + 1.0))
    degree_norm = normalize_prior_matrix(degree_mix)

    drug_similarity = torch.as_tensor(data['drs'], dtype=torch.float32)
    disease_similarity = torch.as_tensor(data['dis'], dtype=torch.float32)
    drug_similarity.fill_diagonal_(0.0)
    disease_similarity.fill_diagonal_(0.0)

    disease_pos_counts = train_drdi.sum(dim=0, keepdim=True).clamp_min(1.0)
    drug_pos_counts = train_drdi.sum(dim=1, keepdim=True).clamp_min(1.0)
    drug_support = (drug_similarity @ train_drdi) / disease_pos_counts
    disease_support = (train_drdi @ disease_similarity) / drug_pos_counts
    collab_norm = normalize_prior_matrix(0.5 * (drug_support + disease_support))

    train_assoc_norm = normalize_prior_matrix(train_drdi) if train_drdi.max() > 0 else train_drdi
    indirect_prior = normalize_prior_matrix(0.50 * shared_norm + 0.35 * collab_norm + 0.15 * degree_norm)
    train_prior = normalize_prior_matrix(
        (1.0 - args.direct_train_prior_weight) * indirect_prior + args.direct_train_prior_weight * train_assoc_norm
    )
    return indirect_prior, train_prior


def gather_pair_bias(pair_index, prior_matrix, device, scale=0.22):
    idx = pair_index.long().detach().cpu().clone()
    idx[:, 0] = idx[:, 0].clamp(0, prior_matrix.shape[0] - 1)
    idx[:, 1] = idx[:, 1].clamp(0, prior_matrix.shape[1] - 1)
    bias = prior_matrix[idx[:, 0], idx[:, 1]].to(device)
    return scale * bias.unsqueeze(-1)


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
    parser.add_argument('--pair_decoder', choices=['hybrid_mlp', 'elementwise'], default='hybrid_mlp')
    parser.add_argument('--path_bias_scale', type=float, default=0.18)
    parser.add_argument('--direct_train_prior_weight', type=float, default=0.18)
    parser.add_argument('--eval_path_bias', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--aux_warmup_epochs', type=int, default=180)
    parser.add_argument('--ranking_weight', type=float, default=0.06)
    parser.add_argument('--ranking_margin', type=float, default=0.18)
    parser.add_argument('--ranking_samples', type=int, default=2048)
    parser.add_argument('--hard_negative_weight', type=float, default=0.04)
    parser.add_argument('--hard_negative_ratio', type=float, default=0.15)
    parser.add_argument('--hard_negative_margin', type=float, default=0.10)
    parser.add_argument('--label_smoothing', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--log_best_only', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    apply_dataset_preset(args)
    args.direct_train_prior_weight = min(max(args.direct_train_prior_weight, 0.0), 1.0)
    args.cl_min_scale = min(max(args.cl_min_scale, 0.0), 1.0)
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
    data = k_fold(data, args)

    if args.fold_indices is None:
        selected_folds = list(range(args.k_fold))
    else:
        selected_folds = list(dict.fromkeys(args.fold_indices))
        invalid = [i for i in selected_folds if i < 0 or i >= args.k_fold]
        if invalid:
            raise ValueError(f'Invalid fold indices {invalid}; valid range is 0..{args.k_fold - 1}')

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(args.device)
    didi_graph = didi_graph.to(args.device)

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
        cross_entropy = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][fold_idx], args)
        drdipr_graph = drdipr_graph.to(args.device)
        train_positive_edges = positive_training_edges(data['X_train'][fold_idx], data['Y_train'][fold_idx])
        eval_prior, train_prior = build_path_prior(data, train_positive_edges, args)
        eval_prior = eval_prior.to(args.device)
        train_prior = train_prior.to(args.device)

        best_metrics = None
        best_auc = -1.0
        no_improve_epochs = 0
        ema_state_dict = None

        for epoch in range(args.epochs):
            model.train()
            cl_weight = contrastive_weight_for_epoch(epoch, args)
            ranking_weight, hard_neg_weight = aux_loss_weights(epoch, args)
            train_edge_stats = {
                'pair_bias': gather_pair_bias(x_train, train_prior, args.device, scale=args.path_bias_scale)
            }
            _, train_score, cl_loss = model(
                drdr_graph,
                didi_graph,
                drdipr_graph,
                drug_feature,
                disease_feature,
                protein_feature,
                drug_topo_feat,
                disease_topo_feat,
                x_train,
                edge_stats=train_edge_stats,
            )
            ce_loss = cross_entropy(train_score, y_train)
            ranking_loss = pair_ranking_loss(train_score, y_train, args.ranking_margin, args.ranking_samples)
            hard_neg_loss = hard_negative_mining_loss(
                train_score, y_train, top_ratio=args.hard_negative_ratio, margin=args.hard_negative_margin
            )
            train_loss = ce_loss + cl_weight * cl_loss + ranking_weight * ranking_loss + hard_neg_weight * hard_neg_loss

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
                    drdr_graph,
                    didi_graph,
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

            if auc > best_auc + 1e-6:
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
            if (not args.log_best_only) or best_mark:
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
