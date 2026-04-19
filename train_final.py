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
    return 0.65 * ce_loss + 0.35 * focal_loss


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
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
    parser.add_argument('--warmup_epochs', default=150, type=int, help='epochs to train before enabling focal/ranking-heavy fine-tune')
    parser.add_argument('--eval_start_epoch', default=50, type=int, help='minimum epochs before evaluation begins')
    parser.add_argument('--score_every', default=10, type=int, help='evaluate every N epochs after eval start')
    parser.add_argument('--log_every', default=25, type=int, help='print training loss every N epochs')
    parser.add_argument('--focal_gamma', default=1.2, type=float, help='focal loss gamma during early training')
    parser.add_argument('--focal_gamma_warm', default=2.0, type=float, help='focal loss gamma during late training')
    parser.add_argument('--contrastive_weight', default=0.08, type=float, help='weight of node-level cross-view contrastive loss')
    parser.add_argument('--contrastive_temperature', default=0.2, type=float, help='temperature for contrastive loss')
    parser.add_argument('--ranking_weight', default=0.12, type=float, help='weight of pairwise ranking loss')
    parser.add_argument('--ranking_margin', default=0.2, type=float, help='margin used in ranking loss')
    parser.add_argument('--ranking_samples', default=2048, type=int, help='maximum positive-negative pairs used in ranking loss')
    parser.add_argument('--hard_negative_weight', default=2.0, type=float, help='reweight difficult negatives based on current positive score')
    parser.add_argument('--label_smoothing', default=0.02, type=float, help='label smoothing for cross entropy')
    parser.add_argument('--patience', default=120, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--target_auc', default=0.96, type=float, help=argparse.SUPPRESS)
    parser.add_argument('--target_auc_warmup', default=400, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--target_auc_patience', default=4, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--plateau_patience', default=3, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--plateau_factor', default=0.5, type=float, help=argparse.SUPPRESS)

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
    print('Early stopping is disabled; training will run for the full epoch budget.')

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drug_view_graphs, disease_view_graphs, data = dgl_similarity_view_graphs(data, args)
    drug_view_graphs = {name: graph.to(device) for name, graph in drug_view_graphs.items()}
    disease_view_graphs = {name: graph.to(device) for name, graph in disease_view_graphs.items()}

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    metric_header = 'Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc'
    AUCs, AUPRs, Accs, Precs, Recs, F1s, MCCs, Epochs = [], [], [], [], [], [], [], []

    for i in range(args.k_fold):
        print(f'\n--- Fold: {i} ---')
        print(metric_header)

        model = AMNTDDA(args).to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = build_scheduler(optimizer, args)

        best_auc = -1.0
        best_metrics = None
        best_state_dict = None

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
        drdipr_graph, data = dgl_heterograph(data, train_positive_edges, args)
        drdipr_graph = drdipr_graph.to(device)

        start = timeit.default_timer()

        for epoch in range(args.epochs):
            model.train()
            _, train_score, aux_losses = model(
                drug_view_graphs,
                disease_view_graphs,
                drdipr_graph,
                drug_feature,
                disease_feature,
                protein_feature,
                X_train,
                return_aux=True,
            )

            use_focal = (epoch + 1) > args.warmup_epochs
            focal_objective = warm_focal_criterion if use_focal else focal_criterion
            classification_loss = weighted_classification_loss(
                train_score,
                Y_train,
                class_weights,
                focal_objective,
                args.label_smoothing,
                args.hard_negative_weight,
                use_focal,
            )
            ranking_loss = pair_ranking_loss(train_score, Y_train, args.ranking_margin, args.ranking_samples)
            contrastive_loss = aux_losses['contrastive']
            train_loss = classification_loss + args.ranking_weight * ranking_loss + args.contrastive_weight * contrastive_loss

            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                elapsed = timeit.default_timer() - start
                print(
                    f'Epoch {epoch + 1:4d} | {elapsed:7.2f}s | loss {train_loss.item():.5f} | '
                    f'cls {classification_loss.item():.5f} | rank {ranking_loss.item():.5f} | '
                    f'ctr {contrastive_loss.item():.5f} | lr {scheduler.get_last_lr()[0]:.6e}'
                )

            should_score = (epoch + 1) >= args.eval_start_epoch and ((epoch + 1 - args.eval_start_epoch) % max(1, args.score_every) == 0)
            if should_score:
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
                    )

                test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
                test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
                AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_pred, test_prob)

                if AUC > best_auc:
                    best_auc = AUC
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_metrics = (AUC, AUPR, accuracy, precision, recall, f1, mcc, epoch + 1)
                    torch.save(model.state_dict(), os.path.join(args.result_dir, f'best_model_fold_{i}.pth'))

                time_now = timeit.default_timer() - start
                best_mark = ' [BEST]' if abs(AUC - best_auc) < 1e-12 else ''
                print(
                    f'Epoch {epoch+1:4d} | {time_now:7.2f}s | '
                    f'AUC {AUC:.5f} | AUPR {AUPR:.5f} | ACC {accuracy:.5f} | '
                    f'P {precision:.5f} | R {recall:.5f} | F1 {f1:.5f} | MCC {mcc:.5f}{best_mark}'
                )

        if best_metrics is None:
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
                )
            test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
            test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
            best_metrics = (*get_metric(Y_test, test_pred, test_prob), args.epochs)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        AUCs.append(best_metrics[0])
        AUPRs.append(best_metrics[1])
        Accs.append(best_metrics[2])
        Precs.append(best_metrics[3])
        Recs.append(best_metrics[4])
        F1s.append(best_metrics[5])
        MCCs.append(best_metrics[6])
        Epochs.append(best_metrics[7])
        if best_state_dict is not None:
            torch.save(best_state_dict, os.path.join(args.result_dir, f'best_model_fold_{i}_cpu.pth'))
        print(f'Fold {i} summary -> best AUC {best_metrics[0]:.5f} at epoch {best_metrics[7]}')

        del model, optimizer, scheduler, drdipr_graph
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(len(AUCs))],
        'Best_Epoch': Epochs,
        'AUC': AUCs,
        'AUPR': AUPRs,
        'Accuracy': Accs,
        'Precision': Precs,
        'Recall': Recs,
        'F1-score': F1s,
        'Mcc': MCCs,
    })

    metrics_only = results_df.drop(columns=['Fold', 'Best_Epoch'])
    summary_df = pd.DataFrame(
        [['Mean', ''] + metrics_only.mean().tolist(), ['Std', ''] + metrics_only.std().tolist()],
        columns=results_df.columns,
    )
    final_df = pd.concat([results_df, summary_df], ignore_index=True)

    print('\n' + '=' * 30 + '\nFINAL RESULTS SUMMARY (IMPROVED PIPELINE)\n' + '=' * 30)
    print(final_df.iloc[-2:])

    csv_path = os.path.join(args.result_dir, '10_fold_results_improved.csv')
    final_df.to_csv(csv_path, index=False)
    print(f'\nSaved improved results to: {csv_path}')
