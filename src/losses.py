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
# Modifications to the original work have been made by Henry Addison
# to allow for sampling

"""All functions related to loss computation and optimization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from pytorch_msssim import ms_ssim

logger = logging.getLogger(__name__)

from .utils import get_score_fn, get_model_fn, is_main_process
from .sde_lib import VESDE, VPSDE
#====================================================================
def get_deterministic_loss_fn(train, reduce_mean=True):
    logger.info(" >> >> INSIDE get_deterministic_loss_fn")
    
    def loss_fn(model, batch, cond, generator=None):
        """Compute the loss function for a deterministic run.

        Args:
        model: A score model.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.
        generator: An optional random number generator so can control the timesteps and initial noise samples used by loss function [ignored in train mode]

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        # for deterministic model, do not use the time or target inputs - set to 0 always
        x = torch.zeros_like(batch)
        t = torch.zeros(batch.shape[0], device=batch.device)
        pred = model(x, cond, t)
        loss = F.mse_loss(pred, batch, reduction="mean")

        #if is_main_process():
        #    logger.info(f" >> >> INSIDE get_deterministic_loss_fn pred {pred}")
        #    logger.info(f" >> >> INSIDE get_deterministic_loss_fn batch {batch}")
        #    logger.info(f" >> >> INSIDE get_deterministic_loss_fn loss {loss}")
        #    logger.info("_____________________________________________________")
        return loss

    return loss_fn
#====================================================================
def get_deterministic_loss_fn_wmse_msssim(
    train,
    reduce_mean: bool = True,
    alpha: float = 0.007,
    beta: float = 0.048,
    lam: float = 0.158,
    precip_max: float = 100.0,
):
    """Deterministic loss: WMSE + MS-SSIM (Hess & Boers 2022 style).

    The model is trained on *untransformed* precipitation:
        - batch and pred are in physical units (e.g. mm/hr),
          so config.data.target_transform_key should be 'none'.

    Args:
        train: Unused, kept for API compatibility.
        reduce_mean: If True, average WMSE over all elements. If False, WMSE
            is averaged per-sample (spatial dims) and then averaged over batch.
        alpha, beta: Weight parameters in
            w(y) = min(alpha * exp(beta * y), 1).
        lam: Convex weight between WMSE and MS-SSIM, i.e.
             L = lam * WMSE + (1 - lam) * (1 - MS_SSIM).
        precip_max: Maximum precipitation (physical units) used as data_range
            for MS-SSIM.

    Returns:
        A loss_fn(model, batch, cond, generator=None) callable.
    """
    if ms_ssim is None:
        raise RuntimeError(
            "pytorch_msssim.ms_ssim is required for WMSE-MS-SSIM loss "
            "(pip install pytorch-msssim)."
        )

    if is_main_process():
        logger.info(
            f" >> >> WMSE-MS-SSIM loss initialised (lam={lam}, alpha={alpha}, beta={beta})"
        )

    # keep call count inside closure so we can log every 50 steps
    call_state = {"n": 0}

    def loss_fn(model, batch, cond, generator=None):
        """Compute WMSE-MS-SSIM loss for deterministic model.

        Args:
            model: The deterministic model f(x=0, cond, t=0).
            batch: Target precipitation in physical units (e.g. mm/hr).
            cond: Conditioning inputs.
            generator: Unused (for API compatibility).

        Returns:
            Scalar loss (WMSE-MS-SSIM) averaged over the mini-batch.
        """
        call_state["n"] += 1

        # For deterministic model, ignore x and t: always zeros.
        x = torch.zeros_like(batch)
        t = torch.zeros(batch.shape[0], device=batch.device)
        pred = model(x, cond, t)  # same physical space as batch

        # ------------------------------------------------------------------
        # 1) batch and pred are already in physical units (mm/hr)
        # ------------------------------------------------------------------
        target_phys = batch
        pred_phys = pred

        # ------------------------------------------------------------------
        # 2) Weighted MSE term (WMSE)
        #      w(y) = min(alpha * exp(beta * y), 1)
        #      WMSE = mean( w(y_true) * (y_pred - y_true)^2 )
        # ------------------------------------------------------------------
        weights = torch.clamp(alpha * torch.exp(beta * target_phys), max=1.0)
        sq_err = (pred_phys - target_phys) ** 2
        weighted_sq_err = weights * sq_err

        if reduce_mean:
            wmse = torch.mean(weighted_sq_err)
        else:
            # Average over non-batch dims, then over batch.
            # Supports [B, H, W] or [B, C, H, W].
            if weighted_sq_err.ndim == 4:
                spatial_dims = (1, 2, 3)
            elif weighted_sq_err.ndim == 3:
                spatial_dims = (1, 2)
            else:
                spatial_dims = tuple(range(1, weighted_sq_err.ndim))
            wmse_per_sample = weighted_sq_err.mean(dim=spatial_dims)
            wmse = wmse_per_sample.mean()

        # ------------------------------------------------------------------
        # 3) MS-SSIM term
        # ------------------------------------------------------------------
        if target_phys.ndim == 3:
            # [B, H, W] -> [B, 1, H, W]
            target_img = target_phys.unsqueeze(1)
            pred_img = pred_phys.unsqueeze(1)
        elif target_phys.ndim == 4:
            # Assume already [B, C, H, W]
            target_img = target_phys
            pred_img = pred_phys
        else:
            raise ValueError(
                f"Unexpected target tensor shape {target_phys.shape}; "
                "expected [B, H, W] or [B, C, H, W]."
            )

        ms_ssim_val = ms_ssim(
            pred_img,
            target_img,
            data_range=precip_max,
            size_average=True,
        )
        ms_ssim_loss = 1.0 - ms_ssim_val

        # ------------------------------------------------------------------
        # 4) Combine terms: WMSE + MS-SSIM (Hess & Boers hyperparameters)
        # ------------------------------------------------------------------
        loss = lam * wmse + (1.0 - lam) * ms_ssim_loss

        # ------------------------------------------------------------------
        # 5) Log components every 50 steps on main process
        # ------------------------------------------------------------------
        if is_main_process() and (call_state["n"] % 50 == 0):
            logger.info(
                " >> >> INSIDE wmse_msssim loss | step %d: wmse=%.6e, ms_ssim_loss=%.6e, total=%.6e",
                call_state["n"],
                wmse.item(),
                ms_ssim_loss.item(),
                loss.item(),
            )
        return loss

    return loss_fn
#====================================================================
def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, cond, generator=None):
        """Compute the loss function.
        Args:
        model: A score model.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = get_score_fn(sde, model, train=train, continuous=continuous)

        if train:
            t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            z = torch.randn_like(batch)
        else:
            t = torch.rand(batch.shape[0], device=batch.device, generator=generator) * (sde.T - eps) + eps
            z = torch.empty_like(batch).normal_(generator=generator)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, cond, t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn
#====================================================================
def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss

    return loss_fn
#====================================================================
def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                        sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn
#====================================================================
def get_loss(sde, train, config):
    print(" >> >> INSIDE losses.get_loss sde", type(sde), ", config.deterministic", config.deterministic, type(config.deterministic))
    logger.info(" >> >> INSIDE losses.get_loss sde %s, config.deterministic %s %s", type(sde), config.deterministic, type(config.deterministic))
    
    if (config.deterministic or (config.deterministic == 'True')):
        if (config.training.det_loss_type == 'WMSE_MS-SSIM'):

            logger.info("[[[[[[[[[[[[[[[[[[[[[[[ WMSE MS-SSIM ]]]]]]]]]]]]]]]]]]]]]]]")


            assert config.data.predictands.target_transform_keys[0] == "none", f"config.data.target_transform_key must be 'none', Got {config.data.predictands.target_transform_keys}"
            loss_fn = get_deterministic_loss_fn_wmse_msssim(train,
                                                            reduce_mean=config.training.reduce_mean,
                                                            alpha=config.training.wmse_ms_ssim_alpha,
                                                            beta=config.training.wmse_ms_ssim_beta,
                                                            lam=config.training.wmse_ms_ssim_lam) 
        else:
            logger.info("[[[[[[[[[[[[[[[[[[[[[[[ regular MSE ]]]]]]]]]]]]]]]]]]]]]]]")
            loss_fn = get_deterministic_loss_fn(train, reduce_mean=config.training.reduce_mean)
        
    else:
        if config.training.continuous:
            loss_fn = get_sde_loss_fn(sde, 
                                      train,
                                      reduce_mean=config.training.reduce_mean,
                                      continuous=True,
                                      likelihood_weighting=config.training.likelihood_weighting)
        else:
            assert not config.training.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, VESDE):
                loss_fn = get_smld_loss_fn(sde, train, reduce_mean=config.training.reduce_mean)
            elif isinstance(sde, VPSDE):
                loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=config.training.reduce_mean)
            else:
                raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")
    return loss_fn
#====================================================================
# def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, deterministic=False):
#   """Create a one-step training/evaluation function.

#   Args:
#     sde: An `sde_lib.SDE` object that represents the forward SDE.
#     optimize_fn: An optimization function.
#     reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
#     continuous: `True` indicates that the model is defined to take continuous time steps.
#     likelihood_weighting: If `True`, weight the mixture of score matching losses according to
#       https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
#     deterministic: If true, use deterministic mode loss, else use diffusion losses.

#   Returns:
#     A one-step function for training or evaluation.
#   """
#   if deterministic:
#     loss_fn = get_deterministic_loss_fn(train, reduce_mean=reduce_mean)
#   else:
#     if continuous:
#       loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
#                               continuous=True, likelihood_weighting=likelihood_weighting)
#     else:
#       assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
#       if isinstance(sde, VESDE):
#         loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
#       elif isinstance(sde, VPSDE):
#         loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
#       else:
#         raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

#   def step_fn(state, batch, cond, generator=None):
#     """Running one step of training or evaluation.

#     This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
#     for faster execution.

#     Args:
#       state: A dictionary of training information, containing the score model, optimizer,
#        EMA status, and number of optimization steps.
#       batch: A mini-batch of training/evaluation data to model.
#       cond: A mini-batch of conditioning inputs.
#       generator: An optional random number generator so can control the timesteps and initial noise samples used by loss function [ignored in train mode]

#     Returns:
#       loss: The average loss value of this state.
#     """
#     model = state['model']
#     if train:
#       optimizer = state['optimizer']
#       optimizer.zero_grad()
#       loss = loss_fn(model, batch, cond)
#       loss.backward()
#       optimize_fn(optimizer, model.parameters(), step=state['step'])
#       state['step'] += 1
#       state['ema'].update(model.parameters())
#     else:
#       with torch.no_grad():
#         ema = state['ema']
#         ema.store(model.parameters())
#         ema.copy_to(model.parameters())
#         loss = loss_fn(model, batch, cond, generator=generator)
#         ema.restore(model.parameters())

#     return loss

#   return step_fn
#====================================================================
# def optimization_manager(config):
#   """Returns an optimize_fn based on `config`."""

#   def optimize_fn(optimizer, params, step, lr=config.optim.lr,
#                   warmup=config.optim.warmup,
#                   grad_clip=config.optim.grad_clip):
#     """Optimizes with warmup and gradient clipping (disabled if negative)."""
#     if warmup > 0:
#       for g in optimizer.param_groups:
#         g['lr'] = lr * np.minimum(step / warmup, 1.0)
#     if grad_clip >= 0:
#       torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
#     optimizer.step()

#   return optimize_fn