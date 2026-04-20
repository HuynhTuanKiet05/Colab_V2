import argparse
import gc
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--topo_hidden', type=int, default=128)
    parser.add_argument('--gate_mode', choices=['scalar', 'vector'], default='vector')
    parser.add_argument('--gate_bias_init', type=float, default=-2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--ema_decay', type=float, default=0.995)

    args = parser.parse_args()
    apply_dataset_preset(args)
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
    print('Epoch\t\tTime\t\tLR\t\tLoss\t\tCL_Loss\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')

    aucs, auprs, accs, precs, recs, f1s, mccs, epochs = [], [], [], [], [], [], [], []
    global_start = timeit.default_timer()
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for fold_idx in selected_folds:
        print(f'\n--- Fold: {fold_idx} ---')
        model = TMC_AMDGT_RVG(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None if args.disable_scheduler else ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6
        )

        x_train = torch.LongTensor(data['X_train'][fold_idx]).to(args.device)
        y_train = torch.LongTensor(data['Y_train'][fold_idx]).to(args.device).flatten()
        x_test = torch.LongTensor(data['X_test'][fold_idx]).to(args.device)
        y_test = data['Y_test'][fold_idx].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][fold_idx], args)
        drdipr_graph = drdipr_graph.to(args.device)

        best_metrics = None
        ema_state_dict = None

        for epoch in range(args.epochs):
            model.train()
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
            )
            ce_loss = cross_entropy(train_score, y_train)
            train_loss = ce_loss + args.lambda_cl * cl_loss

            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            ema_state_dict = update_ema(ema_state_dict, model.state_dict(), args.ema_decay)

            if (epoch + 1) % max(1, args.score_every) != 0:
                continue

            backup_state = None
            if ema_state_dict is not None:
                backup_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(ema_state_dict, strict=False)
            model.eval()
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
                )
            if backup_state is not None:
                model.load_state_dict(backup_state, strict=False)

            test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
            test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
            auc, aupr, accuracy, precision, recall, f1, mcc = get_metric(y_test, test_pred, test_prob)

            if scheduler is not None:
                scheduler.step(auc)
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = timeit.default_timer() - global_start
            print(
                '\t\t'.join(
                    map(
                        str,
                        [
                            epoch + 1,
                            round(elapsed, 2),
                            f'{current_lr:.1e}',
                            round(float(train_loss.item()), 5),
                            round(float(cl_loss.item()), 5),
                            round(float(auc), 5),
                            round(float(aupr), 5),
                            round(float(accuracy), 5),
                            round(float(precision), 5),
                            round(float(recall), 5),
                            round(float(f1), 5),
                            round(float(mcc), 5),
                        ],
                    )
                )
            )

            if best_metrics is None or auc > best_metrics[0]:
                best_metrics = (auc, aupr, accuracy, precision, recall, f1, mcc, epoch + 1)
                if args.save_checkpoints:
                    state_to_save = ema_state_dict if ema_state_dict is not None else model.state_dict()
                    torch.save(state_to_save, os.path.join(args.result_dir, f'best_model_fold_{fold_idx}.pth'))

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
        csv_name = '10_fold_results_tmc_improved.csv'
    else:
        csv_name = f"selected_fold_results_tmc_improved_{'_'.join(map(str, selected_folds))}.csv"
    csv_path = os.path.join(args.result_dir, csv_name)
    final_df.to_csv(csv_path, index=False)
    print(f'\nSaved TMC results to: {csv_path}')
