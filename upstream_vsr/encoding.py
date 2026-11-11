"""Encoder settings shared by data preparation and the live server.

Training inputs and live streams must come from the same filter graph and x264
settings, or the model sees inputs at run time that it never saw in training.
"""
from __future__ import annotations

# Output size and input size per rendition. 'gt' is the 720p reference in training
# and the direct-delivery stream live; lr2 and lr4 are the x2 and x4 inputs.
SIZES = {'gt': (1280, 720), 'lr2': (640, 360), 'lr4': (320, 180)}

X264 = ['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency', '-threads', '1',
        '-x264-params', 'keyint=30:min-keyint=30:scenecut=0:bframes=0:ref=1',
        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
        '-color_trc', 'bt709', '-color_range', 'tv']

KEYINT = 30

_SCALE = ':flags=area+accurate_rnd:out_color_matrix=bt709:out_range=tv'
# The second scale must be told its input is BT.709. Older ffmpeg (as on Kaggle)
# does not carry the first scale's output matrix forward and assumes BT.601,
# which shifts the colours of the low-resolution inputs but not the ground truth.
_RESCALE = _SCALE + ':in_color_matrix=bt709:in_range=tv'


def frame_step(fps: float) -> int:
    """45+ fps sources are halved so that frame spacing resembles a 30 fps stream."""
    return 2 if fps >= 45 else 1


def filter_graph(fps: float) -> str:
    """ffmpeg -filter_complex producing labelled outputs [gt], [lr2] and [lr4]."""
    step = 'framestep=2,' if frame_step(fps) == 2 else ''
    return (f'[0:v]{step}scale=1280:720{_SCALE},format=yuv444p,split=3[gt][a][b];'
            f'[a]scale=640:360{_RESCALE}[lr2];[b]scale=320:180{_RESCALE}[lr4]')


def ladder_graph(fps: float, copies: dict[str, int]) -> str:
    """filter_graph with each output split into copies, labelled [gt_0], [gt_1], [lr2_0], ...

    Every copy of a rendition must be encoded inside this one graph. Re-encoding the
    lossless reference later in a separate ffmpeg loses its colour matrix and shifts
    the colours by about 5 levels, the same fault as in the Kaggle data.
    A label with 0 copies is discarded.
    """
    graph = filter_graph(fps)
    for label in ('gt', 'lr2', 'lr4'):
        count = copies.get(label, 0)
        graph += f';[{label}]' + (f'split={count}' + ''.join(f'[{label}_{i}]' for i in range(count))
                                   if count else 'nullsink')
    return graph


def x264_live() -> list[str]:
    """X264 plus SPS/PPS before every keyframe, so a client can join or switch there.

    Repeating headers does not change any decoded pixel.
    """
    settings = list(X264)
    index = settings.index('-x264-params') + 1
    settings[index] += ':repeat-headers=1'
    return settings
