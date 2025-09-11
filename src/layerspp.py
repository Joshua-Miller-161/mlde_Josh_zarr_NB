# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
print(" >> >> INSIDE layerspp")
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional

from .layers import ddpm_conv1x1, ddpm_conv3x3, NIN, default_init, Upsample, Downsample
print(" >> >> INSIDE layerspp")
#====================================================================
class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = ddpm_conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

# ----------------------------
# ResNet blocks (DDPM++ & BigGAN++)
# ----------------------------

class ResnetBlockDDPMpp(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: Optional[int] = None, *, act, dropout: float = 0.0,
                 init_scale: float = 1.0, skip_rescale: bool = True, temb_dim: Optional[int] = None):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.in_ch, self.out_ch = in_ch, out_ch
        self.act = act
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip_rescale = skip_rescale

        self.norm1 = nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6, affine=True)
        self.conv1 = ddpm_conv3x3(in_ch, out_ch, bias=True)

        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, out_ch)
            default_init()(self.temb_proj.weight.shape)
            nn.init.zeros_(self.temb_proj.bias)
        else:
            self.temb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=min(max(out_ch // 4, 1), 32), num_channels=out_ch, eps=1e-6, affine=True)
        # Last conv uses init_scale to allow residual-zeroing behavior
        self.conv2 = ddpm_conv3x3(out_ch, out_ch, bias=True, init_scale=init_scale)

        self.skip = nn.Identity() if in_ch == out_ch else ddpm_conv1x1(in_ch, out_ch, bias=True)

    def forward(self, x, temb: Optional[torch.Tensor] = None):
        h = self.conv1(self.act(self.norm1(x)))
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(self.act(temb))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        x_skip = self.skip(x)
        if self.skip_rescale:
            return (x_skip + h) / math.sqrt(2.0)
        else:
            return x_skip + h


class ResnetBlockBigGANpp(nn.Module):
    """
    BigGAN-style block with optional up/down.
    """
    def __init__(self, in_ch: int, out_ch: Optional[int] = None, *, act, dropout: float = 0.0,
                 fir: bool = False, fir_kernel: Optional[list] = None, init_scale: float = 1.0,
                 skip_rescale: bool = True, temb_dim: Optional[int] = None, up: bool = False, down: bool = False):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.in_ch, self.out_ch = in_ch, out_ch
        self.act = act
        self.up = up
        self.down = down
        self.skip_rescale = skip_rescale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm1 = nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=min(max(out_ch // 4, 1), 32), num_channels=out_ch, eps=1e-6, affine=True)

        self.upsample = Upsample(in_ch=in_ch, with_conv=False, fir=fir, fir_kernel=fir_kernel) if up else nn.Identity()
        self.downsample = Downsample(in_ch=out_ch, with_conv=False, fir=fir, fir_kernel=fir_kernel) if down else nn.Identity()

        # Use convs without stride; up/down is handled outside convs to keep parameter counts straightforward
        self.conv1 = ddpm_conv3x3(in_ch, out_ch, bias=True)
        self.conv2 = ddpm_conv3x3(out_ch, out_ch, bias=True, init_scale=init_scale)

        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, out_ch)
            default_init()(self.temb_proj.weight.shape)
            nn.init.zeros_(self.temb_proj.bias)
        else:
            self.temb_proj = None

        # skip path: 1x1 to match channels; apply up/down similarly to main path
        self.skip_proj = None
        if in_ch != out_ch:
            self.skip_proj = ddpm_conv1x1(in_ch, out_ch, bias=True)

    def forward(self, x, temb: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        h = self.act(h)
        if self.up is True:
            x = self.upsample(x)
            h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)

        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(self.act(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.down is True:
            x = self.downsample(x)
            h = F.avg_pool2d(h, kernel_size=2, stride=2)

        x_skip = x if self.skip_proj is None else self.skip_proj(x)
        if self.skip_rescale:
            return (x_skip + h) / math.sqrt(2.0)
        else:
            return x_skip + h