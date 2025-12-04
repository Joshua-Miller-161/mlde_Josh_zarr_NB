import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import math
import functools
import logging
logger = logging.getLogger(__name__)

from ..layers import get_act, ddpm_conv3x3, Upsample, Downsample
from ..layerspp import AttnBlockpp, ResnetBlockBigGANpp, ResnetBlockDDPMpp, Combine
from ..utils import register_model
#from .data_scripts.data_utils import get_variables
from ..data_scripts.collate_np_per_var import get_variables

@register_model(name="det_cncsnpp")
class cNCSNpp(nn.Module):
    """NCSN++ model with conditioning input — pure PyTorch, no custom extensions."""
    def __init__(self, config):
        super().__init__()

        logger.info(" + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
        logger.info(" >> >> INSIDE DET CNCSNPP")
        logger.info(" + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
        
        self.config = config
        self.act = act = get_act(config)
        self.nf = nf = config.model.nf
        ch_mult = list(config.model.ch_mult)
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = set(config.model.attn_resolutions)  # use set for fast lookup
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        fir = config.model.fir
        fir_kernel = getattr(config.model, "fir_kernel", [1, 3, 3, 1])
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = config.model.progressive_input.lower()
        init_scale = config.model.init_scale
        embedding_type = config.model.embedding_type.lower()
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []

        # In deterministic mode we do not supply time/noise embeddings. Keep the
        # architecture identical otherwise by simply disabling the temb path.
        embed_dim = None

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
                temb_dim=embed_dim,
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
                temb_dim=embed_dim,
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

        channels = cond_var_channels + cond_time_channels + config.model.loc_spec_channels # No noise input, only atmospheric vars, dates, and location parameters

        logger.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        logger.info(" >> >> >> INSIDE DET cNCSNpp channels: %d, cond_var_channels: %d, cond_time_channels: %d, output_channels: %d, loc_spec_channels: %d", channels, cond_var_channels, cond_time_channels, output_channels, config.model.loc_spec_channels)
        logger.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

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

    def forward(self, x, cond, time_cond):
        # Combine modeled data and conditioning inputs
        #x = torch.cat([x, cond], dim=1)

        logger.info(f" >> >> INSIDE DET CNCSNPP forward x={x.shape} {type(x)}, cond={cond.shape} {type(cond)}, time_cond={time_cond.shape} {time_cond}")
        x = cond
        modules = self.all_modules
        m_idx = 0

        # Deterministic model ignores diffusion timesteps/noise levels.
        temb = None

        # Progressive input pyramid
        input_pyramid = x if self.config.model.progressive_input != "none" else None

        # Downsampling tower
        #logger.info(f" >> >> INSIDE DET CNCSNPP forward x={x.shape}")
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

        return h
