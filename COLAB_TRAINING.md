# Google Colab Training Guide

Huong dan nay gom dung quy trinh da debug de chay repo `Colab_V2` tren Google Colab Linux.

Tai lieu nay uu tien:

- clone repo sach
- cai moi truong on dinh
- restart runtime dung luc
- smoke test truoc khi train dai
- train bang `train_final.py`
- xu ly cac loi Colab da gap trong qua trinh debug

## 0. Chuan bi

Truoc khi chay, vao:

- `Runtime -> Change runtime type`
- chon `GPU`

Kiem tra nhanh:

```python
!nvidia-smi
import sys, platform
print(sys.version)
print(platform.platform())
```

## 1. Clone repo sach tu dau

```python
%cd /content
!rm -rf /content/Colab_V2
!git clone https://github.com/DucTri2207/Colab_V2.git
%cd /content/Colab_V2
!git log --oneline -1
```

Neu ban da co repo san va chi muon cap nhat:

```python
%cd /content/Colab_V2
!git pull origin main
```

## 2. Cai moi truong

Khong nen dua het vao mot shell script khi runtime Colab dang co nhieu package he thong. Cach on dinh nhat la cai theo tung khoi rieng.

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

Sau khi runtime len lai, chay tiep.

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

Khong dung heredoc `python - <<'PY'` trong notebook neu khong can thiet. Cu import truc tiep trong cell Python de de debug hon.

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

Nen chay smoke test truoc khi train 1000 epoch.

```python
%cd /content/Colab_V2
!python scripts/colab_train.py --dataset C-dataset --preset smoke
```

Neu smoke test pass, moi chay train dai.

## 5. Train ban improved

Day la lenh train theo bo tham so da su dung:

```python
%cd /content/Colab_V2
!python train_final.py \
  --epochs 1000 \
  --k_fold 10 \
  --neighbor 20 \
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --hgt_layer 3 \
  --hgt_in_dim 128 \
  --dataset C-dataset \
  --device cuda
```

## 6. Train va luu ket qua len Google Drive

Nen mount Drive truoc khi chay dai de tranh mat checkpoint khi runtime ngat.

### 6.1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 6.2. Train va ghi ket qua ra Drive

```python
%cd /content/Colab_V2
!python train_final.py \
  --epochs 1000 \
  --k_fold 10 \
  --neighbor 20 \
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --hgt_layer 3 \
  --hgt_in_dim 128 \
  --dataset C-dataset \
  --device cuda \
  --result_root /content/drive/MyDrive/Colab_V2_runs/C-dataset_run1
```

## 7. Neu muon chay ban goc AMDGT

Neu ban muon bam dung script goc `train_DDA.py` thay vi ban improved:

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

## 8. Cac loi da gap va cach xu ly

### 8.1. `No such file or directory: /content/Colab_V2`

Repo chua duoc clone.

Chay lai:

```python
%cd /content
!git clone https://github.com/DucTri2207/Colab_V2.git
%cd /content/Colab_V2
```

### 8.2. `ModuleNotFoundError: No module named 'dgl'`

DGL chua cai hoac cai hong.

Chay lai:

```python
!pip install --no-cache-dir --force-reinstall \
  dgl==2.4.0+cu121 \
  -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

Sau do restart runtime.

### 8.3. `module 'networkx' has no attribute 'from_numpy_matrix'`

Day la loi tuong thich giua code cu va `networkx` moi.

Da duoc fix trong repo. Chi can:

```python
%cd /content/Colab_V2
!git pull origin main
```

### 8.4. `module 'dgl.function' has no attribute 'src_mul_edge'`

Day la loi API DGL cu. Repo da duoc fix de dung DGL 2.x.

Chi can:

```python
%cd /content/Colab_V2
!git pull origin main
```

### 8.5. Warning `Ignoring invalid distribution ~vidia-cublas-cu12`

Thuong la package cu bi doi ten trong runtime Colab. Neu `torch`, `dgl` va smoke test van pass thi co the bo qua.

### 8.6. Warning dependency conflict cua Colab

Colab co rat nhieu package he thong khong lien quan den train model.

Muc tieu khong phai lam toan bo Colab "sach", ma la dam bao cac package can cho train dung version:

- `torch`
- `torchvision`
- `torchaudio`
- `dgl`
- `numpy`
- `pandas`
- `scikit-learn`
- `networkx`
- `torchdata`

Neu cac package nay import duoc va smoke test pass, co the tiep tuc train.

## 9. Thu tu chay khuyen nghi

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
10. Train dai

## 10. Ghi chu quan trong

- Luon `git pull origin main` truoc khi train neu repo da duoc fix them.
- Start voi `C-dataset` truoc.
- `F-dataset` nang hon nhieu, nen uu tien Colab High-RAM.
- Sau moi lan doi `torch`, `numpy`, `dgl`, phai restart runtime.
- Neu smoke test fail, khong nen chay 1000 epoch ngay.

## 11. Nguon install chinh thuc

- PyTorch previous versions: `https://pytorch.org/get-started/previous-versions/`
- DGL wheel index for torch 2.4 / cu121: `https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html`
