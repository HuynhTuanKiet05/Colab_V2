# Google Colab Training

## 1. Clone repo

```bash
!git clone https://github.com/DucTri2207/Colab_V2.git
%cd Colab_V2
```

## 2. Install dependencies

```bash
!bash scripts/colab_setup.sh
```

## 3. Smoke test

```bash
!python scripts/colab_train.py --dataset C-dataset --preset smoke
```

## 4. Train standard run

```bash
!python scripts/colab_train.py --dataset C-dataset --preset standard --mount-drive
```

## 5. Full training

```bash
!python scripts/colab_train.py --dataset C-dataset --preset full --mount-drive
```

## Notes

- `--mount-drive` saves results to `"/content/drive/MyDrive/Colab_V2_runs/<dataset>/<preset>"`.
- If you want to save elsewhere, pass `--result-root`.
- If Colab does not expose a GPU, add `--device cpu`.
- Start with `C-dataset`. `F-dataset` is much heavier and is safer on a High-RAM Colab runtime.
- For direct control, you can still call `train_final.py` yourself:

```bash
!python train_final.py --dataset C-dataset --device cuda --data_root AMDGT_original/data/C-dataset --result_root /content/results/C-dataset
```
