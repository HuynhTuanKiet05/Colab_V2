import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
import os
import gc

# Import from the root modules
from data_preprocess_improved import get_data, data_processing, k_fold, dgl_similarity_graph, dgl_heterograph
from model.improved.improved_model import AMNTDDA
from metric import get_metric

# Set device
# DGL on this Windows setup is built without CUDA support, so we fall back to CPU
# to avoid runtime errors when moving DGL graphs to GPU.
device = torch.device('cpu')


class EarlyStopping:
    def __init__(self, patience=200, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_auc):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=10, help='neighbor (balanced for signal vs cost)')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout')
    parser.add_argument('--warmup_epochs', default=250, type=int, help='epochs to train before switching to focal fine-tune')
    parser.add_argument('--patience', default=120, type=int, help='early stopping patience')
    parser.add_argument('--target_auc', default=0.96, type=float, help='target AUC to keep training toward')
    parser.add_argument('--target_auc_warmup', default=400, type=int, help='minimum epochs before scoring and target AUC stopping can trigger')
    parser.add_argument('--target_auc_patience', default=4, type=int, help='how many evaluation checks without target AUC improvement to tolerate')
    parser.add_argument('--score_every', default=1, type=int, help='evaluate every N epochs after warmup')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='focal loss gamma')
    parser.add_argument('--focal_gamma_warm', default=1.2, type=float, help='focal loss gamma after warmup')
    parser.add_argument('--plateau_patience', default=3, type=int, help='lr plateau patience')
    parser.add_argument('--plateau_factor', default=0.5, type=float, help='lr reduction factor on plateau')

    # Model dimensions
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

    # Setup directories
    args.data_dir = 'AMDGT_original/data/' + args.dataset + '/'
    args.result_dir = 'Result/improved/' + args.dataset + '/'
    os.makedirs(args.result_dir, exist_ok=True)

    print(f"--- Starting Final Improved Pipeline ---")
    print(f"Dataset: {args.dataset} | LR: {args.lr} | Dim: {args.gt_out_dim} | Neighbor: {args.neighbor}")

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    Metric_Header = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs, Accs, Precs, Recs, F1s, MCCs, Epochs = [], [], [], [], [], [], [], []

    for i in range(args.k_fold):
        print(f'\n--- Fold: {i} ---')
        print(Metric_Header)

        model = AMNTDDA(args).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True,
            min_lr=1e-6,
        )
        early_stopping = EarlyStopping(patience=args.patience, verbose=False)

        best_auc = 0
        best_metrics = None
        no_improve_checks = 0
        best_state_dict = None
        best_epoch = 0
        best_auc_epoch = 0
        best_auc_was_updated = False
        moving_auc = None
        auc_ema_beta = 0.92

        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        n_pos = torch.sum(Y_train).item()
        n_neg = Y_train.numel() - n_pos
        alpha = torch.tensor([1.0, max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
        ce_criterion = nn.CrossEntropyLoss(weight=alpha)
        focal_criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
        warm_focal_criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma_warm)

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        gc.collect()
        torch.cuda.empty_cache()

        start = timeit.default_timer()

        for epoch in range(args.epochs):
            model.train()
            _, train_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)
            train_targets = torch.flatten(Y_train)
            if epoch < args.warmup_epochs:
                train_loss = ce_criterion(train_score, train_targets)
            else:
                train_loss = 0.4 * ce_criterion(train_score, train_targets) + 0.6 * warm_focal_criterion(train_score, train_targets)

            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if epoch < args.warmup_epochs and (epoch + 1) % 50 == 0:
                time_now = timeit.default_timer() - start
                print(f'Epoch {epoch+1:4d} | {time_now:7.2f}s | warmup training... loss {train_loss.item():.5f}')

            should_score = epoch + 1 >= args.target_auc_warmup and ((epoch + 1 - args.target_auc_warmup) % args.score_every == 0)
            if should_score:
                model.eval()
                with torch.no_grad():
                    _, test_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test)

                test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
                test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
                AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_pred, test_prob)

                moving_auc = AUC if moving_auc is None else (auc_ema_beta * moving_auc + (1 - auc_ema_beta) * AUC)
                improved = AUC > best_auc
                if improved:
                    best_auc = AUC
                    best_auc_epoch = epoch + 1
                    best_auc_was_updated = True
                    no_improve_checks = 0
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_metrics = (AUC, AUPR, accuracy, precision, recall, f1, mcc, epoch + 1)
                    torch.save(model.state_dict(), os.path.join(args.result_dir, f'best_model_fold_{i}.pth'))
                else:
                    best_auc_was_updated = False
                    if AUC < args.target_auc:
                        no_improve_checks += 1
                    else:
                        no_improve_checks = 0

                early_stopping(AUC)
                scheduler.step(AUC)

                time_now = timeit.default_timer() - start
                best_mark = ' [BEST]' if improved else ''
                print(
                    f'Epoch {epoch+1:4d} | {time_now:7.2f}s | '
                    f'AUC {AUC:.5f} | AUPR {AUPR:.5f} | ACC {accuracy:.5f} | '
                    f'P {precision:.5f} | R {recall:.5f} | F1 {f1:.5f} | MCC {mcc:.5f}{best_mark}'
                )
                print(f'          Best AUC so far: {best_auc:.5f} at epoch {best_auc_epoch} | no-improve checks: {no_improve_checks}/{args.target_auc_patience}')

                if early_stopping.early_stop:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

                if no_improve_checks >= args.target_auc_patience and best_auc < args.target_auc:
                    print(
                        f'Target-AUC stop at epoch {epoch+1}: best AUC has stayed below {args.target_auc:.2f} '
                        f'for {no_improve_checks} evaluation checks.'
                    )
                    break

        if best_metrics:
            AUCs.append(best_metrics[0]); AUPRs.append(best_metrics[1]); Accs.append(best_metrics[2])
            Precs.append(best_metrics[3]); Recs.append(best_metrics[4]); F1s.append(best_metrics[5])
            MCCs.append(best_metrics[6]); Epochs.append(best_metrics[7])
            if best_state_dict is not None:
                torch.save(best_state_dict, os.path.join(args.result_dir, f'best_model_fold_{i}_cpu.pth'))
            print(f'Fold {i} summary -> best AUC {best_metrics[0]:.5f} at epoch {best_metrics[7]}')

        del model, optimizer, scheduler, drdipr_graph
        gc.collect()
        torch.cuda.empty_cache()

    results_df = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(len(AUCs))],
        'Best_Epoch': Epochs,
        'AUC': AUCs, 'AUPR': AUPRs, 'Accuracy': Accs,
        'Precision': Precs, 'Recall': Recs, 'F1-score': F1s, 'Mcc': MCCs
    })

    metrics_only = results_df.drop(columns=['Fold', 'Best_Epoch'])
    summary_df = pd.DataFrame([['Mean', ''] + metrics_only.mean().tolist(), ['Std', ''] + metrics_only.std().tolist()], columns=results_df.columns)
    final_df = pd.concat([results_df, summary_df], ignore_index=True)

    print('\n' + '='*30 + '\nFINAL RESULTS SUMMARY (IMPROVED PIPELINE)\n' + '='*30)
    print(final_df.iloc[-2:])

    csv_path = os.path.join(args.result_dir, '10_fold_results_improved.csv')
    final_df.to_csv(csv_path, index=False)
    print(f'\nKết quả cải tiến đã được lưu tại: {csv_path}')