"""Decode H.264 in-process and expose the motion vectors already in the bitstream."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import av
import numpy as np

CELL = 4  # H.264's smallest motion partition is 4x4 luma pixels.
AV_PICTURE_TYPE_I = 1


class DecodedFrame(NamedTuple):
    rgb: np.ndarray  # (H, W, 3) uint8
    mv: np.ndarray  # (H/4, W/4, 2) float32, input pixels, frame -> reference, (dx, dy)
    is_intra: bool


def rasterize_motion_vectors(vectors: np.ndarray | None, height: int, width: int) -> np.ndarray:
    """Paint libav AVMotionVector records onto a 4x4-cell grid.

    Only past references are used: the live encoder has no B-frames, and a causal
    model cannot use future references anyway. Cells with no vector stay zero.
    """
    rows, cols = height // CELL, width // CELL
    field = np.zeros((rows, cols, 2), dtype=np.float32)
    if vectors is None or len(vectors) == 0:
        return field
    past = vectors[vectors['source'] < 0]
    for w, h, dst_x, dst_y, motion_x, motion_y, motion_scale in zip(
        past['w'], past['h'], past['dst_x'], past['dst_y'],
        past['motion_x'], past['motion_y'], past['motion_scale'],
    ):
        x0 = max((int(dst_x) - int(w) // 2) // CELL, 0)
        y0 = max((int(dst_y) - int(h) // 2) // CELL, 0)
        x1 = min(x0 + max(int(w) // CELL, 1), cols)
        y1 = min(y0 + max(int(h) // CELL, 1), rows)
        scale = max(int(motion_scale), 1)
        field[y0:y1, x0:x1, 0] = int(motion_x) / scale
        field[y0:y1, x0:x1, 1] = int(motion_y) / scale
    return field


def to_decoded(frame: av.VideoFrame) -> DecodedFrame:
    """RGB, motion field and frame type, converted exactly as the training data was."""
    side = frame.side_data.get('MOTION_VECTORS')
    vectors = side.to_ndarray() if side is not None else None
    rgb = frame.reformat(format='rgb24', src_colorspace='ITU709').to_ndarray()
    field = rasterize_motion_vectors(vectors, frame.height, frame.width)
    return DecodedFrame(rgb, field, int(frame.pict_type) == AV_PICTURE_TYPE_I)


def decode_with_motion(path: str | Path) -> Iterator[DecodedFrame]:
    """Yield frames in presentation order with their motion fields."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.options = {'flags2': '+export_mvs'}
        for frame in container.decode(stream):
            yield to_decoded(frame)


class MotionDecoder:
    """Decode H.264 one packet at a time, as the live client receives them.

    Single-threaded on purpose: frame threading would hold frames back. The stream
    must have no B-frames, so every packet yields its frame immediately.
    """

    def __init__(self) -> None:
        self.context = av.CodecContext.create('h264', 'r')
        self.context.options = {'flags2': '+export_mvs'}
        self.context.thread_count = 1

    def decode(self, data: bytes) -> list[DecodedFrame]:
        return [to_decoded(frame) for frame in self.context.decode(av.Packet(data))]
