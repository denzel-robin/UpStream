"""Published super-resolution models, run the way the live client runs UpStreamVSR.

Each baseline takes one decoded frame at a time (causal, no future frames) and returns
the 720p output, so it can replace UpStreamVSR in live/client.py (--upscaler) and in
scripts/benchmark_baselines.py without changing anything else.

  efrlfn   EfRLFN (Bogatyrev et al., ICLR 2026), trained on StreamSR by its authors.
           Single-image, x2 and x4. MIT licence.
  span     SPAN, 48 channels (Wan et al., CVPR 2024 workshops; first in the NTIRE 2024
           efficient SR challenge). Single-image, trained on clean DIV2K/LSDIR images,
           x2 and x4. Apache 2.0 licence.
  nanovsr  NanoVSR-226k (Pawlicki et al., ECCV 2026). Recurrent video model, x4 only.
           It is bidirectional; live, the backward branch sees only the current frame,
           as it does for the last frame of every training sequence. MIT licence.
  cdavsr   CDA-VSR (CVPR 2026). Online video model using codec motion vectors,
           residual maps and frame types, x4 only. mmcv's modulated deformable
           convolution is replaced by torchvision's (same operation and weights).
           The residual map is the motion-compensation error of the decoded frame,
           because the live client has no access to the decoder's residuals.

The architectures below are inference-only re-statements of the authors' code, with
the same parameter names, so the published checkpoints load unchanged (strict).
Weights are read from $UPSTREAM_EXTERNAL (default ~/Github/btp-vsr/external), laid out
as the authors' repositories with the downloaded weights:

  efrlfn/weights/EfRLFN-{2,4}x-model.pt     span/weights/spanx{2,4}_ch48.pth
  nanovsr/checkpoints/nanovsr_226k.pth      cda-vsr/pretrained_models/best.pth
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

EXTERNAL = Path(os.environ.get('UPSTREAM_EXTERNAL', Path.home() / 'Github/btp-vsr/external'))
NAMES = ('efrlfn', 'span', 'nanovsr', 'cdavsr')


def load_weights(path: Path, key: str | None = None) -> dict:
    state = torch.load(path, map_location='cpu', weights_only=True)
    return state[key] if key else state


def conv(cin: int, cout: int, k: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, k, padding=(k - 1) // 2, bias=bias)


def upsampler(cin: int, cout: int, scale: int) -> nn.Sequential:
    return nn.Sequential(conv(cin, cout * scale * scale, 3), nn.PixelShuffle(scale))


# EfRLFN -----------------------------------------------------------------

class ECA(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True).squeeze(-1).permute(0, 2, 1)
        y = torch.sigmoid(self.conv(y).permute(0, 2, 1).unsqueeze(-1))
        return x * y


class ERLFB(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1_r, self.c2_r, self.c3_r = conv(c, c, 3), conv(c, c, 3), conv(c, c, 3)
        self.c5, self.eca = conv(c, c, 1), ECA()

    def forward(self, x):
        out = torch.tanh(self.c3_r(torch.tanh(self.c2_r(torch.tanh(self.c1_r(x))))))
        return self.eca(self.c5(out + x))


class EfRLFN(nn.Module):
    def __init__(self, scale: int, c: int = 52):
        super().__init__()
        self.conv_1 = conv(3, c, 3)
        for i in range(1, 7):
            setattr(self, f'block_{i}', ERLFB(c))
        self.conv_2 = conv(c, c, 3)
        self.upsampler = upsampler(c, 3, scale)

    def forward(self, x):
        feature = self.conv_1(x)
        out = feature
        for i in range(1, 7):
            out = getattr(self, f'block_{i}')(out)
        return self.upsampler(self.conv_2(out + feature))


# SPAN -------------------------------------------------------------------

class Conv3XC(nn.Module):
    """Training-time branches kept only so the checkpoint loads; inference uses the
    fused eval_conv, computed once by fuse()."""

    def __init__(self, cin: int, cout: int, gain: int = 2, relu: bool = False):
        super().__init__()
        self.has_relu = relu
        self.sk = nn.Conv2d(cin, cout, 1)
        self.conv = nn.Sequential(nn.Conv2d(cin, cin * gain, 1), nn.Conv2d(cin * gain, cout * gain, 3),
                                  nn.Conv2d(cout * gain, cout, 1))
        self.eval_conv = nn.Conv2d(cin, cout, 3, padding=1)

    @torch.no_grad()
    def fuse(self) -> None:  # as Conv3XC.update_params in the SPAN repository
        w1, b1 = self.conv[0].weight, self.conv[0].bias
        w2, b2 = self.conv[1].weight, self.conv[1].bias
        w3, b3 = self.conv[2].weight, self.conv[2].bias
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        weight = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3).flip(2, 3).permute(1, 0, 2, 3)
        bias = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        self.eval_conv.weight.copy_(weight + F.pad(self.sk.weight, [1, 1, 1, 1]))
        self.eval_conv.bias.copy_(bias + self.sk.bias)
        del self.sk, self.conv

    def forward(self, x):
        out = self.eval_conv(x)
        return F.leaky_relu(out, 0.05) if self.has_relu else out


class SPAB(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1_r, self.c2_r, self.c3_r = Conv3XC(c, c), Conv3XC(c, c), Conv3XC(c, c)

    def forward(self, x):
        # The original applies SiLU in place, so the out1 it returns is already activated.
        out1 = F.silu(self.c1_r(x))
        out3 = self.c3_r(F.silu(self.c2_r(out1)))
        return (out3 + x) * (torch.sigmoid(out3) - 0.5), out1


class SPAN(nn.Module):
    def __init__(self, scale: int, c: int = 48):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1), persistent=False)
        self.conv_1 = Conv3XC(3, c)
        for i in range(1, 7):
            setattr(self, f'block_{i}', SPAB(c))
        self.conv_cat = conv(c * 4, c, 1)
        self.conv_2 = Conv3XC(c, c)
        self.upsampler = upsampler(c, 3, scale)

    def fuse(self) -> 'SPAN':
        for module in list(self.modules()):
            if isinstance(module, Conv3XC):
                module.fuse()
        return self

    def forward(self, x):
        feature = self.conv_1((x - self.mean) * 255)
        b1, _ = self.block_1(feature)
        out = b1
        for i in range(2, 6):
            out, _ = getattr(self, f'block_{i}')(out)
        b6, b5_2 = self.block_6(out)
        out = self.conv_cat(torch.cat([feature, self.conv_2(b6), b1, b5_2], 1))
        return self.upsampler(out)


# NanoVSR ------------------------------------------------------------------

class RepBlock(nn.Module):
    """RepVGG block: 3x3 + 1x1 + identity, each with BatchNorm; fused to one 3x3."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.rbr_identity = nn.BatchNorm2d(cin) if cin == cout else None
        self.rbr_dense = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.rbr_1x1 = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.rbr_reparam = nn.Conv2d(cin, cout, 3, padding=1)

    @staticmethod
    def _fold(kernel: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        std = (bn.running_var + bn.eps).sqrt()
        return kernel * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * bn.weight / std

    @torch.no_grad()
    def fuse(self) -> None:
        k3, b3 = self._fold(self.rbr_dense[0].weight, self.rbr_dense[1])
        k1, b1 = self._fold(F.pad(self.rbr_1x1[0].weight, [1, 1, 1, 1]), self.rbr_1x1[1])
        kernel, bias = k3 + k1, b3 + b1
        if self.rbr_identity is not None:
            c = self.rbr_identity.num_features
            identity = torch.zeros(c, c, 3, 3, device=kernel.device)
            identity[torch.arange(c), torch.arange(c), 1, 1] = 1
            ki, bi = self._fold(identity, self.rbr_identity)
            kernel, bias = kernel + ki, bias + bi
        self.rbr_reparam.weight.copy_(kernel)
        self.rbr_reparam.bias.copy_(bias)
        del self.rbr_dense, self.rbr_1x1, self.rbr_identity

    def forward(self, x):
        return F.leaky_relu(self.rbr_reparam(x), 0.1)


class NanoShuffle(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class NanoVSR(nn.Module):
    def __init__(self, c: int = 32, blocks: int = 8):
        super().__init__()
        self.feat_extract = RepBlock(3, c)
        self.forward_net = nn.Sequential(*[RepBlock(c, c) for _ in range(blocks)])
        self.backward_net = nn.Sequential(*[RepBlock(c, c) for _ in range(blocks)])
        self.fusion = nn.Conv2d(c * 2, c, 1)
        self.upsample1 = NanoShuffle(c, c)
        self.upsample2 = NanoShuffle(c, 32)
        self.conv_last = nn.Conv2d(32, 3, 3, 1, 1)

    def fuse(self) -> 'NanoVSR':
        for module in list(self.modules()):
            if isinstance(module, RepBlock):
                module.fuse()
        return self

    def step(self, x, state):
        """One frame: the forward branch carries state; the backward branch has no
        future frames, so it starts from zero, as for the last frame of a sequence."""
        feat = self.feat_extract(x)
        forward = self.forward_net(feat + (state if state is not None else 0))
        backward = self.backward_net(feat)
        out = self.conv_last(self.upsample2(self.upsample1(self.fusion(torch.cat([forward, backward], 1)))))
        return out + F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False), forward


# CDA-VSR -----------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, c: int = 64):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(c, c, 3, 1, 1), nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class DeformAlign(nn.Module):
    def __init__(self, c: int = 64, groups: int = 4, max_residue: float = 8):
        super().__init__()
        self.max_residue = max_residue
        self.conv_offset = nn.Sequential(
            nn.Conv2d(c * 3 + 2, c // 4, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(c // 4, c // 4, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(c // 4, c // 4, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(c // 4, 3 * groups, 1))
        self.dcn = nn.Module()
        self.dcn.weight = nn.Parameter(torch.zeros(2 * c, 2 * c, 1, 1))

    def forward(self, x, cond, flow):
        offset1, offset2, mask = torch.chunk(self.conv_offset(torch.cat([cond, flow], 1)), 3, 1)
        offset = self.max_residue * torch.tanh(torch.cat([offset1, offset2], 1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        return deform_conv2d(x, offset, self.dcn.weight, mask=torch.sigmoid(mask))


def mv_warp(x: torch.Tensor, flow: torch.Tensor, mode: str = 'nearest') -> torch.Tensor:
    """Sample x at each pixel plus flow (B, 2, H, W), in pixels, borders clamped."""
    _, _, h, w = x.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
    gx = 2 * (xs + flow[:, 0]) / max(w - 1, 1) - 1
    gy = 2 * (ys + flow[:, 1]) / max(h - 1, 1) - 1
    grid = torch.stack([gx, gy], 3).to(x.dtype)
    return F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)


def luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class CDAVSR(nn.Module):
    def __init__(self, c: int = 64, extract: int = 3, blocks_i: int = 24, blocks_p: int = 12):
        super().__init__()
        self.conv_first = nn.Conv2d(3, c, 3, padding=1)
        self.feature_extraction = nn.Sequential(*[ResBlock(c) for _ in range(extract)])
        self.deform_align = DeformAlign(c)
        self.conv_weight_l = nn.Sequential(nn.Conv2d(1, c, 1), nn.LeakyReLU(0.1), nn.Conv2d(c, c, 1),
                                           nn.LeakyReLU(0.1), nn.Conv2d(c, 1, 1))
        self.convs = nn.Sequential(nn.Conv2d(c * 3, c, 3, padding=1))
        self.reconstruction_I = nn.Sequential(*[ResBlock(c) for _ in range(blocks_i)])
        self.reconstruction_P = nn.Sequential(*[ResBlock(c) for _ in range(blocks_p)])
        self.conv_h = nn.Conv2d(c, c, 3, padding=1)
        self.upconv1 = nn.Conv2d(c, 48, 3, 1, 1)

    def step(self, x, flow, intra: bool, state):
        """flow: per-pixel motion (B, 2, H, W) from this frame to the previous one."""
        feat = self.feature_extraction(self.conv_first(x))
        if intra or state is None:  # frame-type-aware: I-frames take the deeper intra branch
            rec = self.reconstruction_I(feat)
        else:
            feat_l, feat_h, x_prev = state
            lh = torch.cat([feat_l, feat_h], 1)
            warped = mv_warp(lh, flow)
            aligned = self.deform_align(lh, torch.cat([warped, feat], 1), flow)
            residual = (luma(x) - luma(mv_warp(x_prev, flow, 'bilinear'))).abs()
            aligned = aligned * self.conv_weight_l(residual) + aligned
            rec = self.reconstruction_P(self.convs(torch.cat([aligned, feat], 1)))
        out = F.pixel_shuffle(self.upconv1(rec), 4)
        return out + F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False), (feat, self.conv_h(rec), x)


# Common interface ----------------------------------------------------------

class Baseline:
    """step(x, mv, is_intra, scale) -> output, for one frame; reset() between streams."""

    def __init__(self, name: str, device: torch.device, half: bool = False, root: Path = EXTERNAL):
        self.name, self.device = name, device
        self.dtype = torch.float16 if half else torch.float32
        self.state = None
        self.last_scale = None
        if name == 'efrlfn':
            self.nets = {s: EfRLFN(s) for s in (2, 4)}
            for s, net in self.nets.items():
                net.load_state_dict(load_weights(root / f'efrlfn/weights/EfRLFN-{s}x-model.pt'))
        elif name == 'span':
            self.nets = {}
            for s in (2, 4):
                net = SPAN(s)
                net.load_state_dict(load_weights(root / f'span/weights/spanx{s}_ch48.pth', 'params_ema'))
                self.nets[s] = net.fuse()
        elif name == 'nanovsr':
            net = NanoVSR()
            state = {k: v for k, v in load_weights(root / 'nanovsr/checkpoints/nanovsr_226k.pth').items()
                     if not k.endswith('num_batches_tracked')}
            missing, unexpected = net.load_state_dict(state, strict=False)
            if unexpected or any('rbr_reparam' not in k for k in missing):
                raise ValueError(f'NanoVSR checkpoint does not match: {missing} {unexpected}')
            self.nets = {4: net.fuse()}
        elif name == 'cdavsr':
            net = CDAVSR()
            net.load_state_dict(load_weights(root / 'cda-vsr/pretrained_models/best.pth', 'params_ema'))
            self.nets = {4: net}
        else:
            raise ValueError(f'unknown baseline {name!r}; choose from {NAMES}')
        for s in self.nets:
            self.nets[s] = self.nets[s].eval().to(device, self.dtype)
        self.scales = tuple(sorted(self.nets))

    @property
    def parameters(self) -> dict[int, int]:
        return {s: sum(p.numel() for p in net.parameters()) for s, net in self.nets.items()}

    def reset(self) -> None:
        self.state, self.last_scale = None, None

    @torch.inference_mode()
    def step(self, x: torch.Tensor, mv: torch.Tensor, is_intra: bool, scale: int) -> torch.Tensor:
        """x (1, 3, h, w) in [0, 1]; mv (1, 2, h/4, w/4), frame -> previous frame, pixels."""
        if scale not in self.nets:
            raise ValueError(f'{self.name} has no x{scale} model')
        if scale != self.last_scale:
            self.state = None  # recurrent state belongs to one resolution
        self.last_scale = scale
        net, x = self.nets[scale], x.to(self.dtype)
        if self.name == 'nanovsr':
            out, self.state = net.step(x, self.state)
        elif self.name == 'cdavsr':
            flow = F.interpolate(mv.to(self.dtype), scale_factor=4, mode='nearest')
            out, self.state = net.step(x, flow, is_intra, self.state)
        else:
            out = net(x)
        return out.clamp(0, 1)
