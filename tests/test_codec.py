"""Checks motion vector direction against a real x264 encode of known motion."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.codec import decode_with_motion  # noqa: E402
from upstream_vsr.model import warp  # noqa: E402

SHIFT = 2  # the texture moves 2 px to the right every frame


@pytest.fixture(scope='module')
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp('codec') / 'moving.mp4'
    size = 320, 180
    rng = np.random.default_rng(0)
    coarse = rng.integers(0, 255, (size[1] // 8 + 1, (size[0] + SHIFT * 40) // 8 + 1, 3), dtype=np.uint8)
    texture = np.kron(coarse, np.ones((8, 8, 1), np.uint8))
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{size[0]}x{size[1]}', '-r', '30', '-i', '-', '-c:v', 'libx264',
               '-preset', 'veryfast', '-tune', 'zerolatency', '-crf', '18',
               '-x264-params', 'keyint=10:min-keyint=10:scenecut=0:bframes=0:ref=1',
               '-pix_fmt', 'yuv420p', str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(20):
        start = SHIFT * (40 - index)
        process.stdin.write(texture[:size[1], start:start + size[0]].tobytes())
    process.stdin.close()
    assert process.wait() == 0
    return path


def test_frame_types_follow_the_gop(clip):
    frames = list(decode_with_motion(clip))
    assert len(frames) == 20
    assert [i for i, frame in enumerate(frames) if frame.is_intra] == [0, 10]


def test_motion_field_shape_and_magnitude(clip):
    frames = list(decode_with_motion(clip))
    assert frames[1].mv.shape == (45, 80, 2)
    # Frame content moved right, so each block came from SHIFT px to its left.
    assert np.median(frames[1].mv[..., 0]) == pytest.approx(-SHIFT, abs=.25)
    assert np.median(frames[1].mv[..., 1]) == pytest.approx(0, abs=.25)
    assert not frames[0].mv.any()


def test_warping_with_codec_vectors_reduces_error(clip):
    frames = list(decode_with_motion(clip))
    previous = torch.from_numpy(frames[4].rgb).permute(2, 0, 1)[None].float()
    current = torch.from_numpy(frames[5].rgb).permute(2, 0, 1)[None].float()
    mv = torch.from_numpy(frames[5].mv).permute(2, 0, 1)[None]
    flow = torch.nn.functional.interpolate(mv, scale_factor=4, mode='nearest')
    warped_error = (warp(previous, flow) - current).abs().mean()
    assert warped_error < (previous - current).abs().mean() * .5
