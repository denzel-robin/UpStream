"""Training windows prepared by scripts/prepare_streamsr.py."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def augment_window(arrays: dict[str, np.ndarray], hflip: bool, vflip: bool, transpose: bool) -> dict[str, np.ndarray]:
    """Geometric augmentation that keeps motion vectors consistent with the pixels.

    Time is never reversed: the model is causal and vectors point backwards.
    """
    out = dict(arrays)
    for key in ('gt', 'lr2', 'lr4', 'mv2', 'mv4'):
        value = out[key]  # (T, H, W, C)
        is_mv = key.startswith('mv')
        if hflip:
            value = value[:, :, ::-1].copy()
            if is_mv:
                value[..., 0] *= -1
        if vflip:
            value = value[:, ::-1].copy()
            if is_mv:
                value[..., 1] *= -1
        if transpose:
            value = value.transpose(0, 2, 1, 3).copy()
            if is_mv:
                value = value[..., ::-1].copy()
        out[key] = value
    return out


class WindowDataset(Dataset):
    def __init__(self, root: str | Path, num_frames: int = 12, train: bool = True):
        self.paths = sorted(Path(root).glob('*.npz'))
        if not self.paths:
            raise FileNotFoundError(f'no prepared windows in {root}')
        self.num_frames = num_frames
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.paths[index]) as window:
            arrays = {key: window[key] for key in window.files}
        total = len(arrays['gt'])
        length = min(self.num_frames, total)
        start = random.randint(0, total - length) if self.train else 0
        for key in ('gt', 'lr2', 'lr4', 'mv2', 'mv4', 'intra2', 'intra4'):
            arrays[key] = arrays[key][start:start + length]
        if self.train:
            arrays = augment_window(arrays, random.random() < .5, random.random() < .5, random.random() < .5)
        sample: dict[str, torch.Tensor] = {}
        for key in ('gt', 'lr2', 'lr4'):
            sample[key] = torch.from_numpy(arrays[key].copy()).permute(0, 3, 1, 2).float() / 255
        for key in ('mv2', 'mv4'):
            sample[key] = torch.from_numpy(arrays[key].astype(np.float32).copy()).permute(0, 3, 1, 2)
        for key in ('intra2', 'intra4'):
            sample[key] = torch.from_numpy(arrays[key].astype(np.float32).copy())
        return sample
