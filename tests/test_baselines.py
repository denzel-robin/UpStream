import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.baselines import EXTERNAL, NAMES, Baseline  # noqa: E402

WEIGHTS = {'efrlfn': 'efrlfn/weights/EfRLFN-4x-model.pt', 'span': 'span/weights/spanx4_ch48.pth',
           'nanovsr': 'nanovsr/checkpoints/nanovsr_226k.pth', 'cdavsr': 'cda-vsr/pretrained_models/best.pth'}


@pytest.mark.parametrize('name', NAMES)
def test_baseline_upscales_frame_by_frame(name):
    if not (EXTERNAL / WEIGHTS[name]).exists():
        pytest.skip(f'{name} weights not downloaded')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Baseline(name, device)
    for scale in model.scales:
        model.reset()
        x = torch.rand(1, 3, 32, 48, device=device)
        mv = torch.zeros(1, 2, 8, 12, device=device)
        for intra in (True, False, False):  # recurrent models carry state across frames
            out = model.step(x, mv, intra, scale)
            assert out.shape == (1, 3, 32 * scale, 48 * scale)
            assert 0 <= out.min() and out.max() <= 1
    assert set(model.scales) == ({4} if name in ('nanovsr', 'cdavsr') else {2, 4})
