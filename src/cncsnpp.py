# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Modifications copyright 2024 Henry Addison
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
#
# Significant modifications to the original work have been made by Henry Addison
# to allow for conditional modelling

# pylint: skip-file
# cncsnpp_vanilla.py
# Pure-PyTorch version of cNCSN++ with no C++/.cu extensions.
# Keeps the same module topology and learnable parameter counts.


# ----------------------------
# cNCSN++ Model (vanilla)
# ----------------------------
import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import math
import functools
import logging
logger = logging.getLogger(__name__)

from .layers import get_act, get_timestep_embedding, default_init, ddpm_conv3x3, Upsample, Downsample
from .layerspp import GaussianFourierProjection, AttnBlockpp, ResnetBlockBigGANpp, ResnetBlockDDPMpp, Combine
from .utils import get_sigmas, register_model
#from .data_scripts.data_utils import get_variables
from .data_scripts.collate_np_per_var import get_variables

@register_model(name="cncsnpp")
class cNCSNpp(nn.Module):
    """NCSN++ model with conditioning input — pure PyTorch, no custom extensions."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = list(config.model.ch_mult)
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = set(config.model.attn_resolutions)  # use set for fast lookup
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = getattr(config.model, "fir_kernel", [1, 3, 3, 1])
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = config.model.progressive_input.lower()
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # Timestep / noise-level embedding
        if embedding_type == "fourier":
            assert config.training.continuous, "Fourier features are only used for continuous training."
            modules.append(GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale))
            embed_dim = 2 * nf
        elif embedding_type == "positional":
            embed_dim = nf
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample_ = functools.partial(Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == "output_skip":
            self.pyramid_upsample = Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)  # in_ch won’t be used
        elif progressive == "residual":
            pyramid_upsample = functools.partial(Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample_ = functools.partial(Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == "input_skip":
            self.pyramid_downsample = Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)  # in_ch won’t be used
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPMpp,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGANpp,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Channel bookkeeping
        # Regular data
        # cond_var_channels, output_channels = list(map(len, get_variables(config.data.dataset_name)))

        # Per var
        cond_var_channels, output_channels = list(map(len, get_variables(config)))
        if config.data.time_inputs:
            cond_time_channels = 3
        else:
            cond_time_channels = 0

        channels = cond_var_channels + cond_time_channels + output_channels + config.model.loc_spec_channels

        logger.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        logger.info(" >> >> >> INSIDE cNCSNpp channels: %d, cond_var_channels: %d, cond_time_channels: %d, output_channels: %d, loc_spec_channels: %d", channels, cond_var_channels, cond_time_channels, output_channels, config.model.loc_spec_channels)
        logger.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        if progressive_input != "none":
            input_pyramid_ch = channels

        # Stem
        modules.append(ddpm_conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf

        # Downsampling tower
        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in self.attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, down=True))
                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2
                elif progressive_input == "residual":
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch
                hs_c.append(in_ch)

        # Bottleneck
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0

        # Upsampling tower
        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in self.attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6))
                        modules.append(ddpm_conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6))
                        modules.append(ddpm_conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                else:
                    if progressive == "output_skip":
                        modules.append(nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6))
                        modules.append(ddpm_conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch))
                        pyramid_ch = in_ch

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample_(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6))
            modules.append(ddpm_conv3x3(in_ch, channels, init_scale=init_scale))

        # Final head (maps back to output channels)
        modules.append(ddpm_conv3x3(channels, output_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.embedding_type = embedding_type

    def forward(self, x, cond, time_cond):
        # Combine modeled data and conditioning inputs
        x = torch.cat([x, cond], dim=1)

        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == "fourier":
            used_sigmas = time_cond  # expected to be continuous sigma (or log) per original
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == "positional":
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f"Unknown embedding type {self.embedding_type}.")

        # Project embedding if conditional
        if self.config.model.conditional:
            temb = modules[m_idx](temb); m_idx += 1
            temb = modules[m_idx](self.act(temb)); m_idx += 1
        else:
            temb = None

        # Progressive input pyramid
        input_pyramid = x if self.config.model.progressive_input != "none" else None

        # Downsampling tower
        hs = [modules[m_idx](x)]; m_idx += 1
        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb); m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h); m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1]); m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb); m_idx += 1

                if self.config.model.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h); m_idx += 1
                elif self.config.model.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid); m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / math.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        # Bottleneck
        h = hs[-1]
        h = modules[m_idx](h, temb); m_idx += 1
        h = modules[m_idx](h); m_idx += 1
        h = modules[m_idx](h, temb); m_idx += 1

        pyramid = None

        # Upsampling tower
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb); m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h); m_idx += 1

            if self.config.model.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.config.model.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h)); m_idx += 1
                        pyramid = modules[m_idx](pyramid); m_idx += 1
                    elif self.config.model.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h)); m_idx += 1
                        pyramid = modules[m_idx](pyramid); m_idx += 1
                else:
                    if self.config.model.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h)); m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h); m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.config.model.progressive == "residual":
                        pyramid = modules[m_idx](pyramid); m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / math.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h); m_idx += 1
                else:
                    h = modules[m_idx](h, temb); m_idx += 1

        assert not hs

        if self.config.model.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h)); m_idx += 1
            h = modules[m_idx](h); m_idx += 1

        # Final head to outputs
        h = modules[m_idx](h); m_idx += 1
        assert m_idx == len(modules)

        if getattr(self.config.model, "scale_by_sigma", False):
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * (x.ndim - 1))))
            h = h / used_sigmas
        return h


# # ----------------------------
# # Utilities & registration
# # ----------------------------

# _MODELS = {}

# def register_model(cls=None, *, name=None):
#     def _register(cls):
#         local_name = name if name is not None else cls.__name__
#         if local_name in _MODELS:
#             raise ValueError(f"Already registered model with name: {local_name}")
#         _MODELS[local_name] = cls
#         return cls
#     return _register if cls is None else _register(cls)

# def get_model(name):
#     return _MODELS[name]

# def default_init_(tensor: torch.Tensor, scale: float = 1.0, nonlinearity: str = "leaky_relu"):
#     # Kaiming init (fan_in) is typical for NCSN++ codebases; multiply by scale for last convs if requested.
#     if tensor.ndim >= 2:
#         nn.init.kaiming_normal_(tensor, a=0.2 if nonlinearity == "leaky_relu" else 0.0, mode="fan_in", nonlinearity=nonlinearity)
#         if scale != 1.0:
#             with torch.no_grad():
#                 tensor.mul_(scale)
#     else:
#         nn.init.zeros_(tensor)

# def get_sigmas(config):
#     return torch.exp(torch.linspace(math.log(config.model.sigma_max), math.log(config.model.sigma_min), config.model.num_scales))

# def get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_positions: int = 10000):
#     """
#     Standard sinusoidal positional embeddings from DDPM/NCSN++.
#     timesteps: (B,)
#     returns: (B, dim)
#     """
#     half = dim // 2
#     freqs = torch.exp(-math.log(max_positions) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
#     args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
#     emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
#     if dim % 2 == 1:
#         emb = F.pad(emb, (0, 1))
#     return emb

# def get_act(config):
#     name = getattr(config.model, "nonlinearity", "swish").lower()
#     if name in ("silu", "swish"):
#         return nn.SiLU()
#     if name in ("relu",):
#         return nn.ReLU(inplace=True)
#     if name in ("elu",):
#         return nn.ELU(inplace=True)
#     if name in ("lrelu", "leaky_relu", "leaky-relu"):
#         negative_slope = getattr(config.model, "lrelu_slope", 0.2)
#         return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
#     raise ValueError(f"Unknown activation: {name}")

# def get_variables(dataset_name: str, config=None) -> Tuple[list, list]:
#     """
#     Return (conditioning variable names list, output variable names list).
#     To keep this file self-contained (no external mlde_josh_utils), read from config if provided.
#     Set:
#       config.data.cond_var_channels  (int)
#       config.data.output_channels    (int)
#     """
#     if config is not None and hasattr(config, "data"):
#         if hasattr(config.data, "cond_var_channels") and hasattr(config.data, "output_channels"):
#             return list(range(int(config.data.cond_var_channels))), list(range(int(config.data.output_channels)))
#     # Fallback: minimal stub; you can expand this mapping if desired.
#     raise ValueError(
#         "Please set 'config.data.cond_var_channels' and 'config.data.output_channels' "
#         "to the correct integers for your dataset."
#     )


# # ----------------------------
# # Small layer helpers
# # ----------------------------

# def conv3x3(in_ch, out_ch, bias=True, init_scale=1.0):
#     m = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
#     default_init_(m.weight, scale=init_scale)
#     if m.bias is not None:
#         nn.init.zeros_(m.bias)
#     return m

# def conv1x1(in_ch, out_ch, bias=True, init_scale=1.0):
#     m = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)
#     default_init_(m.weight, scale=init_scale)
#     if m.bias is not None:
#         nn.init.zeros_(m.bias)
#     return m


# # ----------------------------
# # Attention block (single-head)
# # ----------------------------

# class AttnBlockpp(nn.Module):
#     def __init__(self, channels: int, skip_rescale: bool = True, init_scale: float = 1.0):
#         super().__init__()
#         self.skip_rescale = skip_rescale
#         self.norm = nn.GroupNorm(num_groups=min(max(channels // 4, 1), 32), num_channels=channels, eps=1e-6, affine=True)
#         self.q = conv1x1(channels, channels, bias=True)
#         self.k = conv1x1(channels, channels, bias=True)
#         self.v = conv1x1(channels, channels, bias=True)
#         self.proj = conv1x1(channels, channels, bias=True, init_scale=init_scale)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         h_ = self.norm(x)

#         q = self.q(h_).reshape(b, c, h * w).permute(0, 2, 1)   # (B, HW, C)
#         k = self.k(h_).reshape(b, c, h * w)                    # (B, C, HW)
#         attn = torch.bmm(q, k) * (1.0 / math.sqrt(c))          # (B, HW, HW)
#         attn = attn.softmax(dim=-1)

#         v = self.v(h_).reshape(b, c, h * w).permute(0, 2, 1)   # (B, HW, C)
#         out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
#         out = self.proj(out)

#         if self.skip_rescale:
#             return (x + out) / math.sqrt(2.0)
#         else:
#             return x + out


# # ----------------------------
# # (Up|Down)sample blocks
# # ----------------------------

# class Upsample(nn.Module):
#     def __init__(self, in_ch: int, with_conv: bool = True, fir: bool = False, fir_kernel: Optional[list] = None):
#         super().__init__()
#         self.with_conv = with_conv
#         self.conv = conv3x3(in_ch, in_ch) if with_conv else nn.Identity()
#         # fir / fir_kernel retained for interface compatibility; implemented via conv if desired.
#         self.fir = fir
#         self.register_buffer("fir_kernel", torch.tensor(fir_kernel, dtype=torch.float32) if fir_kernel is not None else None)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         x = self.conv(x)
#         return x

# class Downsample(nn.Module):
#     def __init__(self, in_ch: int, with_conv: bool = True, fir: bool = False, fir_kernel: Optional[list] = None):
#         super().__init__()
#         self.with_conv = with_conv
#         # Using stride-2 conv when with_conv to match usual param counts/behavior
#         self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=True) if with_conv else nn.Identity()
#         if with_conv:
#             default_init_(self.conv.weight)
#             if self.conv.bias is not None:
#                 nn.init.zeros_(self.conv.bias)
#         self.fir = fir
#         self.register_buffer("fir_kernel", torch.tensor(fir_kernel, dtype=torch.float32) if fir_kernel is not None else None)

#     def forward(self, x):
#         if self.with_conv:
#             return self.conv(x)
#         else:
#             return F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)


# # ----------------------------
# # Combine helper (skip pathways)
# # ----------------------------

# class Combine(nn.Module):
#     def __init__(self, method: str = "cat", dim1: int = 0, dim2: int = 0):
#         super().__init__()
#         self.method = method
#         self.dim1 = dim1
#         self.dim2 = dim2

#     def forward(self, x1, x2):
#         if self.method == "cat":
#             return torch.cat([x1, x2], dim=1)
#         elif self.method in ("sum", "add"):
#             return x1 + x2
#         else:
#             raise ValueError(f"Unknown combine method: {self.method}")


# # ----------------------------
# # ResNet blocks (DDPM++ & BigGAN++)
# # ----------------------------

# class ResnetBlockDDPMpp(nn.Module):
#     def __init__(self, in_ch: int, out_ch: Optional[int] = None, *, act, dropout: float = 0.0,
#                  init_scale: float = 1.0, skip_rescale: bool = True, temb_dim: Optional[int] = None):
#         super().__init__()
#         out_ch = in_ch if out_ch is None else out_ch
#         self.in_ch, self.out_ch = in_ch, out_ch
#         self.act = act
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.skip_rescale = skip_rescale

#         self.norm1 = nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6, affine=True)
#         self.conv1 = conv3x3(in_ch, out_ch, bias=True)

#         if temb_dim is not None:
#             self.temb_proj = nn.Linear(temb_dim, out_ch)
#             default_init_(self.temb_proj.weight)
#             nn.init.zeros_(self.temb_proj.bias)
#         else:
#             self.temb_proj = None

#         self.norm2 = nn.GroupNorm(num_groups=min(max(out_ch // 4, 1), 32), num_channels=out_ch, eps=1e-6, affine=True)
#         # Last conv uses init_scale to allow residual-zeroing behavior
#         self.conv2 = conv3x3(out_ch, out_ch, bias=True, init_scale=init_scale)

#         self.skip = nn.Identity() if in_ch == out_ch else conv1x1(in_ch, out_ch, bias=True)

#     def forward(self, x, temb: Optional[torch.Tensor] = None):
#         h = self.conv1(self.act(self.norm1(x)))
#         if self.temb_proj is not None and temb is not None:
#             h = h + self.temb_proj(self.act(temb))[:, :, None, None]
#         h = self.conv2(self.dropout(self.act(self.norm2(h))))
#         x_skip = self.skip(x)
#         if self.skip_rescale:
#             return (x_skip + h) / math.sqrt(2.0)
#         else:
#             return x_skip + h


# class ResnetBlockBigGANpp(nn.Module):
#     """
#     BigGAN-style block with optional up/down.
#     """
#     def __init__(self, in_ch: int, out_ch: Optional[int] = None, *, act, dropout: float = 0.0,
#                  fir: bool = False, fir_kernel: Optional[list] = None, init_scale: float = 1.0,
#                  skip_rescale: bool = True, temb_dim: Optional[int] = None, up: bool = False, down: bool = False):
#         super().__init__()
#         out_ch = in_ch if out_ch is None else out_ch
#         self.in_ch, self.out_ch = in_ch, out_ch
#         self.act = act
#         self.up = up
#         self.down = down
#         self.skip_rescale = skip_rescale
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#         self.norm1 = nn.GroupNorm(num_groups=min(max(in_ch // 4, 1), 32), num_channels=in_ch, eps=1e-6, affine=True)
#         self.norm2 = nn.GroupNorm(num_groups=min(max(out_ch // 4, 1), 32), num_channels=out_ch, eps=1e-6, affine=True)

#         self.upsample = Upsample(in_ch=in_ch, with_conv=False, fir=fir, fir_kernel=fir_kernel) if up else nn.Identity()
#         self.downsample = Downsample(in_ch=out_ch, with_conv=False, fir=fir, fir_kernel=fir_kernel) if down else nn.Identity()

#         # Use convs without stride; up/down is handled outside convs to keep parameter counts straightforward
#         self.conv1 = conv3x3(in_ch, out_ch, bias=True)
#         self.conv2 = conv3x3(out_ch, out_ch, bias=True, init_scale=init_scale)

#         if temb_dim is not None:
#             self.temb_proj = nn.Linear(temb_dim, out_ch)
#             default_init_(self.temb_proj.weight)
#             nn.init.zeros_(self.temb_proj.bias)
#         else:
#             self.temb_proj = None

#         # skip path: 1x1 to match channels; apply up/down similarly to main path
#         self.skip_proj = None
#         if in_ch != out_ch:
#             self.skip_proj = conv1x1(in_ch, out_ch, bias=True)

#     def forward(self, x, temb: Optional[torch.Tensor] = None):
#         h = self.norm1(x)
#         h = self.act(h)
#         if self.up is True:
#             x = self.upsample(x)
#             h = F.interpolate(h, scale_factor=2, mode="nearest")
#         h = self.conv1(h)

#         if self.temb_proj is not None and temb is not None:
#             h = h + self.temb_proj(self.act(temb))[:, :, None, None]

#         h = self.norm2(h)
#         h = self.act(h)
#         h = self.dropout(h)
#         h = self.conv2(h)
#         if self.down is True:
#             x = self.downsample(x)
#             h = F.avg_pool2d(h, kernel_size=2, stride=2)

#         x_skip = x if self.skip_proj is None else self.skip_proj(x)
#         if self.skip_rescale:
#             return (x_skip + h) / math.sqrt(2.0)
#         else:
#             return x_skip + h


# # ----------------------------
# # Gaussian Fourier features (for continuous-time training)
# # ----------------------------

# class GaussianFourierProjection(nn.Module):
#     def __init__(self, embedding_size: int = 256, scale: float = 1.0):
#         super().__init__()
#         # Fixed random weights
#         self.register_buffer("W", torch.randn(embedding_size) * scale)

#     def forward(self, x: torch.Tensor):
#         # x: log(sigma) typically, shape (B,)
#         x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
#         return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

