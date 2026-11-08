"""UpStreamVSR network.

Design sources (re-implemented here, see docs/MODEL.md for the full rationale):

* NanoVSR (Pawlicki et al., ECCV 2026): reparameterizable blocks, additive
  propagation of recurrent state, no custom operators.
* CDA-VSR (CVPR 2026): codec motion vectors to align recurrent state, a
  residual-driven gate on temporal fusion, frame-type awareness, and a cheap
  pixel-shuffle head that runs at the low-resolution grid.

Additions in this project:

* Anytime depth: shared backbone with several exits. The exit is chosen per
  frame at run time with no reload, and recurrent state stays valid across exits.
* Resolution-switchable state: x2 and x4 inputs map onto one feature grid
  (output size / 4), so both modes cost about the same and state survives a
  resolution switch. Convolutions are shared between scales but BatchNorm is
  per scale, because x2 and x4 features have different statistics.

The model is strictly causal: each output depends only on current and past frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALES = (2, 4)


class RepBlock(nn.Module):
    """3x3 + 1x1 + identity branches at training time; one plain 3x3 conv once fused.

    Convolutions are shared across input scales, but each scale has its own
    BatchNorm. x2 and x4 inputs have different feature statistics, so a single
    set of running statistics fits neither: it follows whichever scale the last
    training batches used. Once fused, each scale has its own plain 3x3 conv.
    """

    def __init__(self, in_channels: int, out_channels: int, scales: Sequence[int] = SCALES):
        super().__init__()
        self.scales = tuple(scales)
        keys = [f'x{scale}' for scale in self.scales]
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn3 = nn.ModuleDict({key: nn.BatchNorm2d(out_channels) for key in keys})
        self.bn1 = nn.ModuleDict({key: nn.BatchNorm2d(out_channels) for key in keys})
        self.bn_id = (nn.ModuleDict({key: nn.BatchNorm2d(in_channels) for key in keys})
                      if in_channels == out_channels else None)
        self.fused: nn.ModuleDict | None = None
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        key = f'x{scale}'
        if self.fused is not None:
            return self.activation(self.fused[key](x))
        out = self.bn3[key](self.conv3(x)) + self.bn1[key](self.conv1(x))
        if self.bn_id is not None:
            out = out + self.bn_id[key](x)
        return self.activation(out)

    @staticmethod
    def _fuse_branch(kernel: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * scale, bn.bias - bn.running_mean * bn.weight / std

    @torch.no_grad()
    def fuse(self) -> None:
        if self.fused is not None:
            return
        fused = nn.ModuleDict()
        for key in self.bn3:
            kernel, bias = self._fuse_branch(self.conv3.weight, self.bn3[key])
            kernel1, bias1 = self._fuse_branch(F.pad(self.conv1.weight, [1, 1, 1, 1]), self.bn1[key])
            kernel, bias = kernel + kernel1, bias + bias1
            if self.bn_id is not None:
                identity = torch.zeros_like(self.conv3.weight)
                for channel in range(identity.size(0)):
                    identity[channel, channel, 1, 1] = 1
                kernel_id, bias_id = self._fuse_branch(identity, self.bn_id[key])
                kernel, bias = kernel + kernel_id, bias + bias_id
            conv = nn.Conv2d(self.conv3.in_channels, self.conv3.out_channels, 3, padding=1, bias=True)
            conv.to(kernel.device, kernel.dtype)
            conv.weight.copy_(kernel)
            conv.bias.copy_(bias)
            fused[key] = conv
        self.fused = fused
        del self.conv3, self.bn3, self.conv1, self.bn1, self.bn_id


def upgrade_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load checkpoints from before per-scale BatchNorm by copying each shared BN to every scale.

    Running statistics copied this way are a compromise between scales; recalibrate
    them with scripts/calibrate_bn.py.
    """
    expected = model.state_dict()
    upgraded = {}
    for name, value in state.items():
        if name in expected:
            upgraded[name] = value
            continue
        prefix, _, field = name.rpartition('.')
        copies = [f'{prefix}.{key}.{field}' for key in ('x2', 'x4') if f'{prefix}.{key}.{field}' in expected]
        if not copies:
            raise KeyError(f'unexpected parameter {name}')
        for copy in copies:
            upgraded[copy] = value.clone()
    return upgraded


@dataclass
class StreamState:
    """Recurrent state. Independent of the input scale, so it survives a switch.

    hidden and previous live on the shared feature grid; output is the last
    reconstructed frame at output resolution.
    """

    hidden: torch.Tensor
    previous: torch.Tensor
    output: torch.Tensor


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample x at (position + flow). flow is (B, 2, H, W) in pixels, order (dx, dy)."""
    _, _, h, w = x.shape
    ys = torch.arange(h, dtype=x.dtype, device=x.device)
    xs = torch.arange(w, dtype=x.dtype, device=x.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    norm_x = 2 * (grid_x + flow[:, 0]) / max(w - 1, 1) - 1
    norm_y = 2 * (grid_y + flow[:, 1]) / max(h - 1, 1) - 1
    grid = torch.stack((norm_x, norm_y), dim=-1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)


class UpStreamVSR(nn.Module):
    """One frame in, one frame out, with explicit recurrent state.

    Args:
        num_feat: feature channels on the grid.
        exits: cumulative block counts, one per anytime exit, ascending.
    """

    def __init__(self, num_feat: int = 32, exits: Sequence[int] = (3, 6, 9)):
        super().__init__()
        exits = tuple(exits)
        if not exits or list(exits) != sorted(set(exits)) or exits[0] < 1:
            raise ValueError('exits must be strictly ascending positive block counts')
        self.num_feat = num_feat
        self.exits = exits
        self.entry = nn.ModuleDict({
            'x2': RepBlock(3 * 4, num_feat, scales=(2,)),  # pixel-unshuffled onto the grid
            'x4': RepBlock(3, num_feat, scales=(4,)),
        })
        self.gate = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        # Start with the gate mostly open, so the model has to learn when to distrust
        # the past instead of starting from a closed gate that gets no useful gradient.
        nn.init.constant_(self.gate[2].bias, 2.0)
        # Current features, warped hidden state and the warped previous output
        # (space-to-depth onto the grid) are mixed by one cheap 1x1 conv.
        self.fusion = nn.Conv2d(2 * num_feat + 48, num_feat, 1)
        self.blocks = nn.ModuleList(RepBlock(num_feat, num_feat) for _ in range(exits[-1]))
        self.exit_adapters = nn.ModuleList(nn.Conv2d(num_feat, num_feat, 1) for _ in exits)
        for adapter in self.exit_adapters:
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
        self.heads = nn.ModuleDict({
            'x2': nn.Sequential(nn.Conv2d(num_feat, 48, 3, padding=1), nn.PixelShuffle(4)),
            'x4': nn.Sequential(nn.Conv2d(num_feat, 48, 3, padding=1), nn.PixelShuffle(4)),
        })
        # Start exactly at bilinear, so training only ever has to add detail.
        for head in self.heads.values():
            nn.init.zeros_(head[0].weight)
            nn.init.zeros_(head[0].bias)

    def step(
        self,
        x: torch.Tensor,
        mv: torch.Tensor,
        is_intra: torch.Tensor,
        state: StreamState | None,
        scale: int,
        exit_index: int = -1,
        all_exits: bool = False,
    ) -> tuple[torch.Tensor | list[torch.Tensor], StreamState]:
        """Reconstruct one frame.

        Args:
            x: decoded frame (B, 3, H, W) in [0, 1]. x2 needs H and W divisible by 2.
            mv: codec motion field (B, 2, H/4, W/4) in input pixels, pointing from
                this frame to its reference, in (dx, dy) order. Zero for intra blocks.
            is_intra: (B,) bool or float, 1 for I-frames.
            state: state from the previous frame, or None at stream start.
            scale: 2 or 4. May change between frames while keeping the state.
            exit_index: which anytime exit to use; its features and output become the state.
            all_exits: also return every other exit's output (for training).

        Returns:
            (output, new_state), where output is (B, 3, H*scale, W*scale), or a list
            with one output per exit when all_exits is set.
        """
        if scale == 4:
            grid_x, feat = x, self.entry['x4'](x, 4)
            flow = F.interpolate(mv, scale_factor=4, mode='nearest')
        elif scale == 2:
            grid_x = F.avg_pool2d(x, 2)
            feat = self.entry['x2'](F.pixel_unshuffle(x, 2), 2)
            flow = F.interpolate(mv, scale_factor=2, mode='nearest') * 0.5
        else:
            raise ValueError(f'scale must be one of {SCALES}, got {scale}')

        b, _, h, w = feat.shape
        intra = is_intra.to(x.dtype).view(b, 1, 1, 1).expand(b, 1, h, w)
        base = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        if state is None:
            warped_hidden = feat.new_zeros(feat.shape)
            warped_previous = grid_x
            warped_output = base
        else:
            warped_hidden = warp(state.hidden, flow)
            warped_previous = warp(state.previous, flow)
            output_flow = F.interpolate(mv, size=base.shape[-2:], mode='nearest') * scale
            warped_output = warp(state.output, output_flow)

        # The motion-compensation error stands in for the codec residual: it is
        # available without a patched decoder and also flags occlusions and bad MVs.
        error = (grid_x - warped_previous).abs().mean(dim=1, keepdim=True)
        gate = self.gate(torch.cat((error, intra), dim=1))
        past = torch.cat((warped_hidden, F.pixel_unshuffle(warped_output, 4)), dim=1)
        out = feat + self.fusion(torch.cat((feat, gate * past), dim=1))

        exit_index %= len(self.exits)
        wanted = range(len(self.exits)) if all_exits else (exit_index,)
        outputs, hidden = {}, None
        for depth, block in enumerate(self.blocks[:self.exits[max(wanted)]], start=1):
            out = block(out, scale)
            if depth in self.exits:
                index = self.exits.index(depth)
                if index in wanted:
                    adapted = out + self.exit_adapters[index](out)
                    outputs[index] = base + self.heads[f'x{scale}'](adapted)
                if index == exit_index:
                    hidden = out
        new_state = StreamState(hidden=hidden, previous=grid_x, output=outputs[exit_index])
        if all_exits:
            return [outputs[index] for index in wanted], new_state
        return outputs[exit_index], new_state

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(upgrade_state_dict(self, state))

    def fuse(self) -> 'UpStreamVSR':
        """Collapse all reparameterizable blocks for deployment."""
        for module in self.modules():
            if isinstance(module, RepBlock):
                module.fuse()
        return self
