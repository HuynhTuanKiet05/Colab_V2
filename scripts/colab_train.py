from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Colab-friendly launcher for train_final.py')
    parser.add_argument('--dataset', default='C-dataset', choices=['B-dataset', 'C-dataset', 'F-dataset'])
    parser.add_argument('--preset', default='standard', choices=['smoke', 'standard', 'full'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--mount-drive', action=argparse.BooleanOptionalAction, default=False, help='mount Google Drive before training')
    parser.add_argument('--drive-root', default='/content/drive/MyDrive/Colab_V2_runs', help='base output folder when using Drive')
    parser.add_argument('--data-root', default=None, help='override dataset directory')
    parser.add_argument('--result-root', default=None, help='override result directory')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--k-fold', type=int, default=None)
    parser.add_argument('--warmup-epochs', type=int, default=None)
    parser.add_argument('--target-auc-warmup', type=int, default=None)
    parser.add_argument('--score-every', type=int, default=None)
    parser.add_argument('--neighbor', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    return parser.parse_args()


def maybe_mount_drive(enabled: bool):
    if not enabled:
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError('--mount-drive chỉ dùng được trong Google Colab.') from exc
    drive.mount('/content/drive', force_remount=False)


def build_preset(args):
    presets = {
        'smoke': {
            'epochs': 2,
            'k_fold': 2,
            'warmup_epochs': 1,
            'target_auc_warmup': 1,
            'score_every': 1,
            'neighbor': 5,
            'lr': 3e-4,
        },
        'standard': {
            'epochs': 180,
            'k_fold': 5,
            'warmup_epochs': 40,
            'target_auc_warmup': 60,
            'score_every': 1,
            'neighbor': 10,
            'lr': 3e-4,
        },
        'full': {
            'epochs': 1000,
            'k_fold': 10,
            'warmup_epochs': 250,
            'target_auc_warmup': 400,
            'score_every': 1,
            'neighbor': 10,
            'lr': 3e-4,
        },
    }
    config = presets[args.preset]
    for key in ['epochs', 'k_fold', 'warmup_epochs', 'target_auc_warmup', 'score_every', 'neighbor', 'lr']:
        override = getattr(args, key)
        if override is not None:
            config[key] = override
    return config


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    maybe_mount_drive(args.mount_drive)

    preset = build_preset(args)
    data_root = Path(args.data_root) if args.data_root else repo_root / 'AMDGT_original' / 'data' / args.dataset
    if args.result_root:
        result_root = Path(args.result_root)
    elif args.mount_drive:
        result_root = Path(args.drive_root) / args.dataset / args.preset
    else:
        result_root = repo_root / 'Result' / 'improved' / args.dataset / f'colab_{args.preset}'

    env = os.environ.copy()
    if args.device == 'auto':
        env.setdefault('AMDGT_DEVICE', 'cuda' if os.environ.get('COLAB_GPU') or os.path.exists('/proc/driver/nvidia/version') else 'cpu')
    else:
        env['AMDGT_DEVICE'] = args.device
    env.setdefault('DGLBACKEND', 'pytorch')

    if args.dataset == 'F-dataset' and args.preset != 'smoke':
        print('Warning: F-dataset uses much more RAM; prefer Colab High-RAM or reduce the preset first.', flush=True)

    cmd = [
        sys.executable,
        'train_final.py',
        '--dataset', args.dataset,
        '--device', args.device,
        '--data_root', str(data_root),
        '--result_root', str(result_root),
        '--epochs', str(preset['epochs']),
        '--k_fold', str(preset['k_fold']),
        '--warmup_epochs', str(preset['warmup_epochs']),
        '--target_auc_warmup', str(preset['target_auc_warmup']),
        '--score_every', str(preset['score_every']),
        '--neighbor', str(preset['neighbor']),
        '--lr', str(preset['lr']),
    ]

    print('Launching training command:', flush=True)
    print(' '.join(cmd), flush=True)
    print(f'AMDGT_DEVICE={env["AMDGT_DEVICE"]}', flush=True)
    print(f'Data root={data_root}', flush=True)
    print(f'Result root={result_root}', flush=True)

    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == '__main__':
    main()
