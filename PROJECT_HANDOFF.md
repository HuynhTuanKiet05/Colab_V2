# Project Handoff

## Muc tieu

Du an nay dang toi uu mo hinh AMDGT cho bai toan du doan tuong tac drug-disease / miRNA-benh.
Rang buoc quan trong:

- Khong sua noi dung trong `AMDGT_original/`.
- Moi cai tien phai nam o cac file improved rieng.
- CSV ket qua phai dung format:
  `Fold,Best_Epoch,AUC,AUPR,Accuracy,Precision,Recall,F1-score,Mcc`
- Dong cuoi phai la `Mean`, dong sau la `Std`, tinh bang `np.mean` va `np.std`.

## Trang thai hien tai

Huong chinh dang dung la `train_tmc_improved.py`, khong phai `train_final.py`.

Ly do:

- `train_final.py` da tung cho mot so y tuong cai tien tot, nhung ket qua khong on dinh tren nhung fold yeu.
- `train_tmc_improved.py` cho ket qua kha thi hon khi tham chieu tu ban zip do an va huong TMC-AMDGT.
- Hien tai `train_tmc_improved.py` da duoc hybrid voi nhung phan huu ich nhat tu `train_final.py`.

## File quan trong

- `AMDGT_original/`
  - Giu nguyen, khong sua.
- `train_final.py`
  - Nhanh improved cu. Van giu de doi chieu y tuong.
- `train_tmc_improved.py`
  - Script train hien dang duoc toi uu va uu tien su dung.
- `model/improved/tmc_rvg_model.py`
  - Model hybrid TMC hien tai.
- `topology_features_improved.py`
  - Trich xuat topology features.
- `COLAB_TRAINING.md`
  - Huong dan chay Colab moi nhat.

## Nhung cai tien dang co trong train_tmc_improved

- Backbone TMC-AMDGT-RVG.
- Topology residual qua `FuzzyGate`.
- Topology feature branch.
- Hybrid pair decoder.
- Ensemble pair decoder:
  - `elementwise` nhanh, on dinh.
  - `hybrid_mlp` giau dac trung.
  - `hybrid_ensemble` hien la mac dinh.
- Train-only path prior.
- Cosine warmup scheduler.
- EMA cho evaluation on dinh hon.
- Class-weighted cross entropy.
- Prior-aware hard sample weighting trong classification loss.
- Contrastive loss co decay theo epoch.
- Ranking loss nhe cho cap kho.
- Hard-negative mining loss nhe.
- Focal loss nhe o nua sau qua trinh train.
- Label smoothing giam dan o cuoi qua trinh train.
- `hybrid_ensemble` gate co nhin thay `pair_bias` khi route hai decoder.
- Chi in log `BEST` theo nguong tang AUC de Colab gon hon.

## Cac commit gan day quan trong

- `90c5996` Add TMC-AMDGT-RVG topology residual training path
- `c30306b` Stabilize TMC training with EMA and smoothing
- `f417af0` Align TMC CSV naming with improved result format
- `6d0b756` Trim TMC training log to report-only metrics
- `02c4ef8` Hybridize TMC training with final decoder and schedule
- `07c0d84` Highlight best TMC validation scores in logs
- `e7e428f` Boost TMC hard-case learning and best-only logging
- `c22c370` Target weak TMC folds with ensemble decoder and focal tuning
- `d0effb0` Throttle BEST log output by AUC gain threshold

Neu can tiep tuc tu chat moi, uu tien doc tu commit moi nhat va cac file ben duoi.

## Cac fold yeu can uu tien

Theo theo doi gan day:

- Fold 1 yeu
- Fold 4 yeu
- Fold 7 yeu

Cap nhat them tu ket qua truoc patch weighting moi:

- Fold 0 = `0.96508`
- Fold 1 = `0.94791`
- Fold 2 = `0.95756`

Suy ra:

- Fold 1 dang la diem nghen lon nhat.
- Fold 2 cung dang duoi muc chap nhan duoc.
- Fold 0 khong te nhat nhung van duoi muc muc tieu `0.97`.
- Neu giu nguyen 3 fold tren thi 7 fold con lai phai dat trung binh khoang `0.97564` moi keo mean 10-fold len `0.97`, nen can uu tien keo manh fold 1 va 2 truoc.

Tham chieu them tu ket qua thay toi uu (anh CSV ngoai repo):

- Mean AUC khoang `0.96930`
- Fold 1 van la diem yeu lon nhat, nhung da duoc keo len khoang `0.95290`
- Fold 2 dat khoang `0.97046`, cao hon ro ret so voi run cu cua nhanh TMC

Phan tich moi nhat:

- Trong `AMDGT_original.data_preprocess`, consensus disease similarity dang co bieu hien sai fallback: khi `DiseasePS == 0` thi dang roi ve `DiseasePS` thay vi `DiseaseGIP`.
- Nhanh improved hien tai khong sua file goc, nhung da duoc bo sung fix o tang improved qua `similarity_fusion_improved.py`.
- `train_tmc_improved.py` hien da recompute `drs/dis` theo `--similarity_fusion nonzero_mean` va dung multi-view collaborative prior thay vi chi dua tren mot consensus matrix.
- `topology_features_improved.py` da doi cache tag mac dinh sang `simfix_v1` de tranh tai lai topology cache cu duoc tinh tu similarity loi.

Muc tieu dang nham:

- Fold 1, 4, 7 can co gang dat tam `>= 0.967`
- Cac fold con lai co gang dat tam `>= 0.970`

## Cach hien thi log hien tai

Mac dinh:

- `--log_best_only`
- `--display_best_delta 0.001`

Nghia la man hinh chi in khi AUC tang du mot nguong.
Vi du:

- `0.96556 -> 0.96590`: khong in
- `0.96556 -> 0.96656`: co in

Dieu nay chi anh huong phan hien thi, khong anh huong ket qua train hay luu checkpoint.

## Lenh full C-dataset dang duoc uu tien

```python
%cd /content/Colab_V2
!git pull origin main
!git log --oneline -1

!python train_tmc_improved.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --neighbor 5 \
  --lr 0.0001 \
  --min_lr 0.000001 \
  --lr_warmup_epochs 40 \
  --weight_decay 0.001 \
  --gt_out_dim 256 \
  --hgt_in_dim 256 \
  --hgt_head_dim 32 \
  --hgt_layer 2 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 2 \
  --tr_layer 2 \
  --tr_head 4 \
  --lambda_cl 0.1 \
  --cl_warmup_epochs 200 \
  --cl_min_scale 0.2 \
  --temperature 0.5 \
  --topo_hidden 128 \
  --gate_mode vector \
  --gate_bias_init -2.0 \
  --pair_decoder hybrid_ensemble \
  --path_bias_scale 0.18 \
  --direct_train_prior_weight 0.18 \
  --no-eval_path_bias \
  --aux_warmup_epochs 180 \
  --ranking_weight 0.06 \
  --ranking_margin 0.18 \
  --ranking_samples 2048 \
  --hard_negative_weight 0.04 \
  --hard_negative_ratio 0.15 \
  --hard_negative_margin 0.10 \
  --focal_weight 0.05 \
  --focal_gamma 1.4 \
  --focal_start_epoch 220 \
  --label_smoothing 0.01 \
  --grad_clip 5.0 \
  --ema_decay 0.995 \
  --display_best_delta 0.001 \
  --log_best_only
```

## Lenh test nhanh fold yeu

```python
!python train_tmc_improved.py \
  --dataset C-dataset \
  --device cuda \
  --k_fold 10 \
  --fold_indices 1 4 7 \
  --epochs 1000 \
  --neighbor 5 \
  --lr 0.0001 \
  --min_lr 0.000001 \
  --lr_warmup_epochs 40 \
  --weight_decay 0.001 \
  --gt_out_dim 256 \
  --hgt_in_dim 256 \
  --hgt_head_dim 32 \
  --hgt_layer 2 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 2 \
  --tr_layer 2 \
  --tr_head 4 \
  --lambda_cl 0.1 \
  --cl_warmup_epochs 200 \
  --cl_min_scale 0.2 \
  --temperature 0.5 \
  --topo_hidden 128 \
  --gate_mode vector \
  --gate_bias_init -2.0 \
  --pair_decoder hybrid_ensemble \
  --path_bias_scale 0.18 \
  --direct_train_prior_weight 0.18 \
  --no-eval_path_bias \
  --aux_warmup_epochs 180 \
  --ranking_weight 0.06 \
  --ranking_margin 0.18 \
  --ranking_samples 2048 \
  --hard_negative_weight 0.04 \
  --hard_negative_ratio 0.15 \
  --hard_negative_margin 0.10 \
  --focal_weight 0.05 \
  --focal_gamma 1.4 \
  --focal_start_epoch 220 \
  --label_smoothing 0.01 \
  --grad_clip 5.0 \
  --ema_decay 0.995 \
  --display_best_delta 0.001 \
  --log_best_only
```

Lenh uu tien tiep theo sau patch weighting moi:

```python
!python train_tmc_improved.py \
  --dataset C-dataset \
  --device cuda \
  --k_fold 10 \
  --fold_indices 0 1 2 \
  --epochs 1000 \
  --neighbor 5 \
  --lr 0.0001 \
  --min_lr 0.000001 \
  --lr_warmup_epochs 40 \
  --weight_decay 0.001 \
  --gt_out_dim 256 \
  --hgt_in_dim 256 \
  --hgt_head_dim 32 \
  --hgt_layer 2 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 2 \
  --tr_layer 2 \
  --tr_head 4 \
  --lambda_cl 0.1 \
  --cl_warmup_epochs 200 \
  --cl_min_scale 0.2 \
  --temperature 0.5 \
  --topo_hidden 128 \
  --gate_mode vector \
  --gate_bias_init -2.0 \
  --pair_decoder hybrid_ensemble \
  --path_bias_scale 0.18 \
  --direct_train_prior_weight 0.18 \
  --ce_hard_negative_weight 0.25 \
  --ce_hard_positive_weight 0.10 \
  --prior_hardness_weight 0.12 \
  --no-eval_path_bias \
  --aux_warmup_epochs 180 \
  --ranking_weight 0.06 \
  --ranking_margin 0.18 \
  --ranking_samples 2048 \
  --hard_negative_weight 0.04 \
  --hard_negative_ratio 0.15 \
  --hard_negative_margin 0.10 \
  --focal_weight 0.05 \
  --focal_gamma 1.4 \
  --focal_start_epoch 220 \
  --label_smoothing 0.01 \
  --grad_clip 5.0 \
  --ema_decay 0.995 \
  --display_best_delta 0.001 \
  --log_best_only
```

## Neu mo chat moi

Chi can noi:

- "Doc `PROJECT_HANDOFF.md` roi tiep tuc toi uu fold 1/4/7"

hoac

- "Doc `PROJECT_HANDOFF.md` va tiep tuc toi uu train_tmc_improved"

la co the vao viec rat nhanh.
