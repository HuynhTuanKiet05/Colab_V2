# Google Colab Training Guide

Huong dan nay da duoc cap nhat theo repo hien tai cua ban:

- Repo: `https://github.com/HuynhTuanKiet05/Colab_V2`
- Nhanh chinh: `main`
- Script train improved: `train_final.py`
- Ho tro chay rieng mot nhom fold qua `--fold_indices`

Tai lieu nay phu hop khi ban dang dung mot tai khoan Colab khac va muon chay lai tu dau.

## 0. Chuan bi

Trong Colab:

- `Runtime -> Change runtime type`
- chon `GPU`

Kiem tra nhanh:

```python
!nvidia-smi
import sys, platform
print(sys.version)
print(platform.platform())
```

## 1. Clone repo sach

Neu ban vua doi tai khoan Colab, hay clone lai repo tu dau:

```python
%cd /content
!rm -rf /content/Colab_V2
!git clone https://github.com/HuynhTuanKiet05/Colab_V2.git
%cd /content/Colab_V2
!git log --oneline -1
```

Neu ban da clone roi va chi muon cap nhat:

```python
%cd /content/Colab_V2
!git pull origin main
!git log --oneline -1
```

## 2. Cai moi truong

Khong nen nhom tat ca vao mot lenh shell dai. Cach on dinh nhat tren Colab la cai theo tung khoi va restart dung luc.

### 2.1. Go cac goi cu

```python
%cd /content/Colab_V2
!pip uninstall -y torch torchvision torchaudio dgl dglgo torchdata numpy pandas scikit-learn networkx
```

### 2.2. Cai PyTorch

```python
!pip install --no-cache-dir --force-reinstall \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 2.3. Restart runtime

```python
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
```

Sau khi runtime len lai, chay tiep tu buoc duoi.

### 2.4. Cai cac package train

```python
%cd /content/Colab_V2
!pip install --no-cache-dir --force-reinstall \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scikit-learn==1.6.1 \
  networkx==3.2.1 \
  torchdata==0.8.0 \
  pyTelegramBotAPI \
  "jedi>=0.19.1"
```

### 2.5. Cai DGL

```python
!pip install --no-cache-dir --force-reinstall \
  dgl==2.4.0+cu121 \
  -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

### 2.6. Restart runtime lan nua

```python
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
```

## 3. Kiem tra moi truong sau khi cai

```python
%cd /content/Colab_V2
import torch, dgl, numpy, pandas, sklearn, networkx, torchdata

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("dgl:", dgl.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("networkx:", networkx.__version__)
print("torchdata:", torchdata.__version__)
```

Neu cell tren chay duoc thi moi truong da on.

## 4. Smoke test truoc khi train dai

```python
%cd /content/Colab_V2
!python scripts/colab_train.py --dataset C-dataset --preset smoke
```

Neu smoke test pass thi moi chay train dai.

## 5. Lenh train khuyen nghi cho C-dataset

Day la bo tham so improved hien tai de train full:

```python
%cd /content/Colab_V2
!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --epochs 1000 \
  --k_fold 10 \
  --neighbor 5 \
  --lr 0.00022 \
  --lr_warmup_epochs 40 \
  --weight_decay 0.00025 \
  --dropout 0.15 \
  --gt_out_dim 256 \
  --hgt_in_dim 256 \
  --hgt_layer 3 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 4 \
  --tr_layer 2 \
  --tr_head 8 \
  --pair_decoder mlp \
  --warmup_epochs 140 \
  --eval_start_epoch 40 \
  --score_every 5 \
  --patience 180 \
  --contrastive_weight 0.025 \
  --ranking_weight 0.14 \
  --path_bias_scale 0.30 \
  --direct_train_prior_weight 0.20 \
  --no-eval_path_bias
```

Ket qua full 10-fold se duoc ghi ra:

- `Result/improved/C-dataset/10_fold_results_improved.csv`

## 5B. Lenh train theo huong TMC-AMDGT-RVG

Day la huong tham khao tu do an ban cua ban: giu backbone AMDGT, bo sung topology qua gated residual.

```python
%cd /content/Colab_V2
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
  --pair_decoder hybrid_mlp \
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
  --label_smoothing 0.01 \
  --grad_clip 5.0 \
  --ema_decay 0.995 \
  --log_best_only
```

Ket qua full 10-fold se duoc ghi ra:

- `Result/tmc_improved/C-dataset/10_fold_results_improved.csv`

## 6. Chay rieng cac fold yeu de tune nhanh

Repo hien tai da ho tro `--fold_indices`.

Neu ban muon chay rieng cac fold:

- `1`
- `2`
- `4`
- `8`

thi dung:

```python
%cd /content/Colab_V2
!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --k_fold 10 \
  --fold_indices 1 2 4 8 \
  --epochs 1000 \
  --neighbor 5 \
  --lr 0.00022 \
  --lr_warmup_epochs 40 \
  --weight_decay 0.00025 \
  --dropout 0.15 \
  --gt_out_dim 256 \
  --hgt_in_dim 256 \
  --hgt_layer 3 \
  --hgt_head 8 \
  --gt_layer 2 \
  --gt_head 4 \
  --tr_layer 2 \
  --tr_head 8 \
  --pair_decoder mlp \
  --warmup_epochs 140 \
  --eval_start_epoch 40 \
  --score_every 5 \
  --patience 180 \
  --contrastive_weight 0.025 \
  --ranking_weight 0.14 \
  --path_bias_scale 0.30 \
  --direct_train_prior_weight 0.20 \
  --no-eval_path_bias
```

Ket qua subset se duoc ghi ra file rieng, vi du:

- `Result/improved/C-dataset/selected_fold_results_improved_1_2_4_8.csv`

Luu y:

- `Mean` va `Std` trong file subset chi tinh tren cac fold duoc chon
- khong dung file subset de bao cao ket qua chinh thuc 10-fold
- no chi dung de tuning nhanh

Neu muon chay rieng fold yeu theo huong TMC-RVG:

```python
%cd /content/Colab_V2
!python train_tmc_improved.py \
  --dataset C-dataset \
  --device cuda \
  --k_fold 10 \
  --fold_indices 1 2 4 8 \
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
  --pair_decoder hybrid_mlp \
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
  --label_smoothing 0.01 \
  --grad_clip 5.0 \
  --ema_decay 0.995 \
  --log_best_only
```

## 7. Luu ket qua len Google Drive

Nen mount Drive truoc khi chay dai de tranh mat ket qua neu runtime bi ngat.

### 7.1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 7.2. Train va luu ket qua ra Drive

```python
%cd /content/Colab_V2
!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --result_root /content/drive/MyDrive/Colab_V2_runs/C-dataset_run1
```

Neu can, ban co the them lai day du bo tham so o Muc 5 hoac Muc 6.

## 8. Neu muon chay ban goc AMDGT

Neu ban muon bam script goc thay vi ban improved:

```python
%cd /content/Colab_V2/AMDGT_original

!python train_DDA.py \
  --epochs 1000 \
  --k_fold 10 \
  --neighbor 20 \
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --hgt_layer 3 \
  --hgt_in_dim 128 \
  --dataset C-dataset
```

## 9. Cac loi da gap va cach xu ly

### 9.1. `No such file or directory: /content/Colab_V2`

Repo chua duoc clone.

```python
%cd /content
!git clone https://github.com/HuynhTuanKiet05/Colab_V2.git
%cd /content/Colab_V2
```

### 9.2. `ModuleNotFoundError: No module named 'dgl'`

DGL chua cai hoac cai hong.

```python
!pip install --no-cache-dir --force-reinstall \
  dgl==2.4.0+cu121 \
  -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

Sau do restart runtime.

### 9.3. `module 'networkx' has no attribute 'from_numpy_matrix'`

Day la loi tuong thich giua code cu va `networkx` moi. Repo hien tai da fix.

```python
%cd /content/Colab_V2
!git pull origin main
```

### 9.4. `module 'dgl.function' has no attribute 'src_mul_edge'`

Day la loi API DGL cu. Repo hien tai da fix de dung DGL 2.x.

```python
%cd /content/Colab_V2
!git pull origin main
```

### 9.5. `RuntimeError: CUDA out of memory`

Neu GPU Colab yeu hon du kien, hay ha kich thuoc model:

```python
%cd /content/Colab_V2
!python train_final.py \
  --dataset C-dataset \
  --device cuda \
  --gt_out_dim 128 \
  --hgt_in_dim 128 \
  --hgt_head 4 \
  --tr_head 4
```

Neu van thieu bo nho, giam tiep:

- `--gt_out_dim 96`
- `--hgt_in_dim 96`
- `--hgt_head 2`
- `--tr_head 2`

### 9.6. Warning `Ignoring invalid distribution ~vidia-cublas-cu12`

Thuong la package cu bi doi ten trong runtime Colab. Neu `torch`, `dgl` va smoke test van pass thi co the bo qua.

### 9.7. Warning dependency conflict cua Colab

Colab co nhieu package he thong khong lien quan den train model.

Muc tieu la dam bao cac package sau dung version:

- `torch`
- `torchvision`
- `torchaudio`
- `dgl`
- `numpy`
- `pandas`
- `scikit-learn`
- `networkx`
- `torchdata`

Neu cac package nay import duoc va smoke test pass thi co the tiep tuc.

## 10. Thu tu chay khuyen nghi

Thu tu dung nhat:

1. Bat GPU runtime
2. Clone repo
3. Cai PyTorch
4. Restart
5. Cai `numpy/pandas/sklearn/networkx/torchdata`
6. Cai DGL
7. Restart
8. Kiem tra version
9. Smoke test
10. Chay subset fold neu dang tune
11. Chay full 10-fold khi da on

## 11. Ghi chu quan trong

- Luon `git pull origin main` truoc khi train.
- Neu ban vua doi tai khoan Colab, clone lai repo sach la an toan nhat.
- `selected_fold_results_improved_...csv` chi dung de tuning, khong dung de nop ket qua chinh thuc.
- Ket qua chinh thuc phai la full `10_fold_results_improved.csv`.
- Sau moi lan doi `torch`, `numpy`, `dgl`, phai restart runtime.
- Neu smoke test fail, khong nen chay 1000 epoch ngay.

## 12. Nguon install chinh thuc

- PyTorch previous versions: `https://pytorch.org/get-started/previous-versions/`
- DGL wheel index for torch 2.4 / cu121: `https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html`
