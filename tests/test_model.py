import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.data import augment_window  # noqa: E402
from upstream_vsr.model import UpStreamVSR, warp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from prepare_streamsr import is_informative  # noqa: E402
from train_upstream import endless  # noqa: E402


def inputs(scale, batch=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    h, w = (32, 48) if scale == 4 else (64, 96)
    x = torch.rand(batch, 3, h, w, generator=generator)
    mv = torch.randn(batch, 2, h // 4, w // 4, generator=generator)
    return x, mv, torch.zeros(batch)


def trained_like_model():
    """A model with non-trivial BatchNorm statistics."""
    torch.manual_seed(0)
    model = UpStreamVSR(num_feat=8, exits=(1, 2)).train()
    for scale in (2, 4):
        for _ in range(3):
            x, mv, intra = inputs(scale)
            state = None
            for exit_index in (0, 1):
                _, state = model.step(x, mv, intra, state, scale, exit_index)
    for head in model.heads.values():  # heads start at zero; give them weights to test through
        torch.nn.init.normal_(head[0].weight, std=.01)
    return model.eval()


def test_untrained_model_outputs_exactly_bilinear():
    model = UpStreamVSR(num_feat=8, exits=(1,)).eval()
    x, mv, intra = inputs(4)
    with torch.no_grad():
        output, _ = model.step(x, mv, intra, None, 4)
    assert torch.equal(output, torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False))


@pytest.mark.parametrize('scale', [2, 4])
def test_output_shape(scale):
    model = UpStreamVSR(num_feat=8, exits=(1, 2))
    x, mv, intra = inputs(scale)
    output, _ = model.step(x, mv, intra, None, scale)
    assert output.shape == (2, 3, x.shape[2] * scale, x.shape[3] * scale)


@pytest.mark.parametrize('scale', [2, 4])
@pytest.mark.parametrize('exit_index', [0, 1])
def test_fused_model_matches_training_structure(scale, exit_index):
    model = trained_like_model()
    x, mv, intra = inputs(scale)
    with torch.no_grad():
        first, state = model.step(x, mv, intra, None, scale, exit_index)
        second, _ = model.step(x, mv, intra, state, scale, exit_index)
        model.fuse()
        fused_first, fused_state = model.step(x, mv, intra, None, scale, exit_index)
        fused_second, _ = model.step(x, mv, intra, fused_state, scale, exit_index)
    assert torch.allclose(first, fused_first, atol=1e-4)
    assert torch.allclose(second, fused_second, atol=1e-4)


def test_past_output_is_unaffected_by_future_frames():
    model = trained_like_model()
    x, mv, intra = inputs(4)
    with torch.no_grad():
        before, state = model.step(x, mv, intra, None, 4)
        future, future_mv, _ = inputs(4, seed=1)
        model.step(future, future_mv, intra, state, 4)
        again, _ = model.step(x, mv, intra, None, 4)
    assert torch.equal(before, again)


def test_state_carries_information():
    model = trained_like_model()
    x, mv, intra = inputs(4)
    with torch.no_grad():
        cold, state = model.step(x, mv, intra, None, 4)
        warm, _ = model.step(x, mv, intra, state, 4)
    assert not torch.allclose(cold, warm)


def test_resolution_switch_and_exit_change_keep_state_shape():
    model = UpStreamVSR(num_feat=8, exits=(1, 2)).eval()
    with torch.no_grad():
        x4, mv4, intra = inputs(4)
        _, state = model.step(x4, mv4, intra, None, 4, 0)
        x2, mv2, intra = inputs(2)
        output, switched = model.step(x2, mv2, intra, state, 2, 1)
    assert switched.hidden.shape == state.hidden.shape
    assert output.shape[-2:] == (128, 192)


def test_equal_output_size_and_grid_for_both_scales():
    model = UpStreamVSR(num_feat=8, exits=(1,)).eval()
    with torch.no_grad():
        x4, mv4, intra = inputs(4)
        x2, mv2, _ = inputs(2)
        out4, state4 = model.step(x4, mv4, intra, None, 4)
        out2, state2 = model.step(x2, mv2, intra, None, 2)
    assert out4.shape == out2.shape
    assert state4.hidden.shape == state2.hidden.shape


def test_warp_moves_content_the_expected_way():
    image = torch.zeros(1, 1, 8, 8)
    image[0, 0, 4, 3] = 1  # bright pixel at x=3, y=4
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = -2  # each output pixel samples 2 px to its left
    assert warp(image, flow)[0, 0, 4, 5] == pytest.approx(1)


def test_invalid_exits_are_rejected():
    with pytest.raises(ValueError):
        UpStreamVSR(exits=(3, 3))


@pytest.mark.parametrize('flags', [(True, False, False), (False, True, False), (False, False, True)])
def test_augmentation_keeps_vectors_consistent_with_pixels(flags):
    """A vector pointing at a marked pixel must still point at it after augmenting."""
    lr4 = np.zeros((1, 16, 16, 3), np.uint8)
    lr4[0, 5, 9, 0] = 255  # reference marker at x=9, y=5
    mv4 = np.zeros((1, 4, 4, 2), np.float32)
    mv4[0, 1, 1] = (5, 0)  # the cell at x=4..7, y=4..7 sees x+5, y+0 -> (9, 5) from (4, 5)
    arrays = {'gt': np.zeros((1, 64, 64, 3), np.uint8), 'lr2': np.zeros((1, 32, 32, 3), np.uint8),
              'lr4': lr4, 'mv2': np.zeros((1, 8, 8, 2), np.float32), 'mv4': mv4}
    out = augment_window(arrays, *flags)
    marker_y, marker_x = map(int, np.argwhere(out['lr4'][0, ..., 0] == 255)[0])
    cell_y, cell_x = map(int, np.argwhere(np.abs(out['mv4'][0]).sum(-1) > 0)[0])
    dx, dy = out['mv4'][0, cell_y, cell_x]
    # Some source pixel inside the moving cell must land exactly on the marker.
    assert cell_x * 4 <= marker_x - dx < (cell_x + 1) * 4
    assert cell_y * 4 <= marker_y - dy < (cell_y + 1) * 4


def test_static_and_blank_windows_are_rejected():
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 255, (1, 16, 16, 3), dtype=np.uint8)
    assert not is_informative(np.repeat(textured, 4, axis=0))  # frozen
    assert not is_informative(np.full((4, 16, 16, 3), 128, np.uint8))  # blank
    assert is_informative(rng.integers(0, 255, (4, 16, 16, 3), dtype=np.uint8))  # changing


def test_endless_loader_starts_a_new_pass_each_epoch():
    passes = []

    class Loader:
        def __iter__(self):
            passes.append(1)
            return iter([len(passes)])

    stream = endless(Loader())
    assert [next(stream) for _ in range(3)] == [1, 2, 3]


def old_layout(state):
    """Collapse per-scale BatchNorm keys into the single-BN layout of earlier checkpoints."""
    old = {}
    for name, value in state.items():
        parts = name.split('.')
        if len(parts) >= 2 and parts[-2] in ('x2', 'x4') and parts[-3] in ('bn3', 'bn1', 'bn_id'):
            old.setdefault('.'.join(parts[:-2] + parts[-1:]), value)
        else:
            old[name] = value
    return old


def test_old_checkpoint_loads_with_shared_statistics():
    source = trained_like_model()
    old = old_layout(source.state_dict())
    assert not any(f'.{bn}.x2.' in name for name in old for bn in ('bn3', 'bn1', 'bn_id'))
    model = UpStreamVSR(num_feat=8, exits=(1, 2))
    model.load_checkpoint_state(old)
    block = model.blocks[0]
    assert torch.equal(block.bn3['x2'].running_mean, block.bn3['x4'].running_mean)
    assert torch.equal(block.bn3['x2'].running_mean, source.blocks[0].bn3['x2'].running_mean)


def test_scales_keep_separate_statistics():
    model = trained_like_model()
    block = model.blocks[0]
    assert not torch.equal(block.bn3['x2'].running_mean, block.bn3['x4'].running_mean)


def test_calibration_only_changes_the_calibrated_scale():
    from calibrate_bn import calibrate

    model = trained_like_model()
    before_x4 = model.blocks[0].bn3['x4'].running_mean.clone()
    before_x2 = model.blocks[0].bn3['x2'].running_mean.clone()
    x, mv, intra = inputs(2, seed=3)
    frames = 3
    batch = {'lr2': x.unsqueeze(1).repeat(1, frames, 1, 1, 1) * 5,
             'mv2': mv.unsqueeze(1).repeat(1, frames, 1, 1, 1),
             'intra2': intra.unsqueeze(1).repeat(1, frames),
             'gt': torch.zeros(2, frames, 3, 128, 192)}
    calibrate(model, [batch], 2, batches=2, device=torch.device('cpu'))
    assert torch.equal(model.blocks[0].bn3['x4'].running_mean, before_x4)
    assert not torch.equal(model.blocks[0].bn3['x2'].running_mean, before_x2)
    assert model.blocks[0].bn3['x2'].momentum == 0.1 and not model.training
