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

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
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
def fit_gamma_params_per_gridpoint(
    data,
    precip_var='precipitation',
    rain_threshold=1.0,
    min_rainy_days=10,
    lat_chunk=10,
    lon_chunk=10,
):
    """Fit per-gridpoint Gamma distribution parameters for the Doury et al. ASYM loss.

    Accepts a dask-backed xr.DataArray, xr.Dataset, or a path to a zarr/netcdf file.
    Data is rechunked to full-time spatial tiles for fitting, then the original
    DataArray passed in is NOT modified (rechunking creates a new lazy view).

    Uses scipy.stats.gamma.fit (floc=0) on rainy-day values at each grid point.
    Falls back to an exponential distribution (shape=1) for grid points with fewer
    than `min_rainy_days` rainy observations.

    A dask LocalCluster is created with workers = min(available_CPUs, 4) for
    overlap of disk I/O and scipy computation. On a 1-CPU machine this reduces to
    a single synchronous worker, which still benefits from the distributed scheduler's
    memory management.

    Args:
        data: dask-backed xr.DataArray (time, lat, lon), xr.Dataset, or str/Path to
              zarr directory or NetCDF file. Variable `precip_var` is extracted if
              a Dataset or file path is given.
        precip_var: Name of the precipitation variable when data is a Dataset or path.
        rain_threshold: Minimum value (same units as data) to classify a timestep as
                        rainy. Default 1.0 (mm/hr or mm/day).
        min_rainy_days: Minimum rainy timesteps required for a successful gamma fit.
                        Grid points below this threshold fall back to shape=1.
        lat_chunk: Number of latitude grid points per spatial tile during fitting.
        lon_chunk: Number of longitude grid points per spatial tile during fitting.

    Returns:
        shape_map (np.ndarray [H, W]): Gamma shape parameter at each grid point.
        scale_map (np.ndarray [H, W]): Gamma scale parameter at each grid point.
    """
    try:
        import xarray as xr
        import dask
        from scipy import stats
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(
            "fit_gamma_params_per_gridpoint requires xarray, dask, scipy, and tqdm. "
            f"Missing: {e}"
        )

    # ------------------------------------------------------------------ #
    # 1. Load / coerce data to xr.DataArray                               #
    # ------------------------------------------------------------------ #
    if isinstance(data, (str, Path)):
        data_path = str(data)
        print(f"[ASYM] Opening precipitation dataset from: {data_path}")
        if data_path.endswith('.zarr') or os.path.isdir(data_path):
            ds = xr.open_zarr(data_path, consolidated=True, chunks='auto')
        else:
            ds = xr.open_dataset(data_path, chunks='auto')
        da = ds[precip_var]
    elif isinstance(data, xr.Dataset):
        da = data[precip_var]
    elif isinstance(data, xr.DataArray):
        da = data
    else:
        raise TypeError(
            f"Expected str, Path, xr.Dataset, or xr.DataArray, got {type(data)}"
        )

    # ------------------------------------------------------------------ #
    # 2. Dimension introspection & original chunk record                   #
    # ------------------------------------------------------------------ #
    if da.ndim != 3:
        raise ValueError(
            f"Expected a 3-D (time, lat, lon) DataArray, got shape {da.shape}"
        )
    time_dim, lat_dim, lon_dim = da.dims
    n_lat = da.sizes[lat_dim]
    n_lon = da.sizes[lon_dim]
    original_chunks = da.chunks  # saved for reference / logging

    print(
        f"[ASYM] DataArray: shape={da.shape}, original chunks={original_chunks}, "
        f"dtype={da.dtype}"
    )

    # ------------------------------------------------------------------ #
    # 3. Rechunk to (all time, lat_chunk, lon_chunk) for tile-wise fitting #
    # ------------------------------------------------------------------ #
    print(f"[ASYM] Rechunking to ({da.sizes[time_dim]}, {lat_chunk}, {lon_chunk}) …")
    da_fit = da.chunk({time_dim: -1, lat_dim: lat_chunk, lon_dim: lon_chunk})
    n_lat_tiles = len(da_fit.chunks[1])
    n_lon_tiles = len(da_fit.chunks[2])
    n_tiles = n_lat_tiles * n_lon_tiles

    # ------------------------------------------------------------------ #
    # 4. Create LocalCluster sized to available CPUs (min 1)              #
    # ------------------------------------------------------------------ #
    n_workers = max(1, min(os.cpu_count() or 1, 4))
    print(f"[ASYM] Starting LocalCluster with {n_workers} worker(s) …")
    try:
        from dask.distributed import LocalCluster, Client
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit='4GB',
            silence_logs=logging.WARNING,
        )
        client = Client(cluster)
        print(f"[ASYM] Dask dashboard: {client.dashboard_link}")
        _use_distributed = True
    except Exception as exc:
        logger.warning("[ASYM] Could not start LocalCluster (%s); using synchronous scheduler.", exc)
        _use_distributed = False

    # ------------------------------------------------------------------ #
    # 5. Tile-by-tile gamma fitting with progress bars                     #
    # ------------------------------------------------------------------ #
    shape_map = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    scale_map = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    n_success = 0
    n_fallback = 0

    from tqdm import tqdm
    pbar_tiles = tqdm(total=n_tiles, desc="Tiles loaded", unit="tile")
    pbar_pts   = tqdm(total=n_lat * n_lon, desc="Gridpoints fitted", unit="pt")

    lat_offsets = np.cumsum([0] + list(da_fit.chunks[1]))
    lon_offsets = np.cumsum([0] + list(da_fit.chunks[2]))

    try:
        for il in range(n_lat_tiles):
            lat_s = int(lat_offsets[il])
            lat_e = int(lat_offsets[il + 1])

            for jl in range(n_lon_tiles):
                lon_s = int(lon_offsets[jl])
                lon_e = int(lon_offsets[jl + 1])

                # Load tile into memory; dask fetches from disk here
                tile_da = da_fit.isel(
                    {lat_dim: slice(lat_s, lat_e), lon_dim: slice(lon_s, lon_e)}
                )
                if _use_distributed:
                    tile = client.compute(tile_da).result()
                    if hasattr(tile, 'values'):
                        tile = tile.values
                else:
                    with dask.config.set(scheduler='synchronous'):
                        tile = tile_da.values  # shape (T, tile_lat, tile_lon)

                pbar_tiles.update(1)

                tile_lat = lat_e - lat_s
                tile_lon = lon_e - lon_s
                inner = tqdm(
                    total=tile_lat * tile_lon,
                    desc=f"Rows {lat_s}–{lat_e - 1}",
                    leave=False,
                )

                for i in range(tile_lat):
                    for j in range(tile_lon):
                        ts = tile[:, i, j].astype(np.float64)
                        rainy = ts[ts > rain_threshold]

                        if len(rainy) < min_rainy_days:
                            shape_map[lat_s + i, lon_s + j] = 1.0
                            scale_map[lat_s + i, lon_s + j] = (
                                float(rainy.mean()) if len(rainy) > 0 else 1.0
                            )
                            n_fallback += 1
                        else:
                            try:
                                shape, _loc, scale = stats.gamma.fit(rainy, floc=0)
                                shape_map[lat_s + i, lon_s + j] = shape
                                scale_map[lat_s + i, lon_s + j] = scale
                                n_success += 1
                            except Exception:
                                shape_map[lat_s + i, lon_s + j] = 1.0
                                scale_map[lat_s + i, lon_s + j] = float(rainy.mean())
                                n_fallback += 1

                        inner.update(1)
                        pbar_pts.update(1)

                inner.close()
    finally:
        pbar_tiles.close()
        pbar_pts.close()
        if _use_distributed:
            client.close()
            cluster.close()

    print(
        f"\n[ASYM] Gamma fitting complete — {n_success} successful fits, "
        f"{n_fallback} fallback (shape=1) fits."
    )
    valid = ~np.isnan(shape_map)
    print(
        f"[ASYM] Shape  : mean={shape_map[valid].mean():.4f}, "
        f"std={shape_map[valid].std():.4f}"
    )
    print(
        f"[ASYM] Scale  : mean={scale_map[valid].mean():.4f}, "
        f"std={scale_map[valid].std():.4f}"
    )
    print(f"[ASYM] NOTE: original DataArray chunking is unchanged ({original_chunks})")

    return shape_map, scale_map
#====================================================================
def get_deterministic_loss_fn_asym_doury(
    train,
    gamma_shape,
    gamma_scale,
    rain_threshold=1.0,
    reduce_mean=True,
):
    """Deterministic loss: Doury et al. (2024) asymmetric MAE (ASYM / Emul-ASYM).

    Implements Eq. 4 from:
        Doury et al. (2024). On the suitability of a convolutional neural network
        based RCM-emulator for fine spatio-temporal precipitation.
        Climate Dynamics, 62, 8591–8618. https://doi.org/10.1007/s00382-024-07350-8

    Loss formula:
        L(y, ŷ) = mean( |y - ŷ|  +  γ²_{i,t} · max(0, y_{i,t} - ŷ_{i,t}) )

    where γ_{i,t} = G_i(y_{i,t}) is the Gamma CDF evaluated at the true
    precipitation value at grid point i and time t, using per-gridpoint
    Gamma(shape_i, scale_i) parameters pre-fitted on the training set.

    The asymmetric term fires only when the model *underestimates* a rainy pixel
    (y > ŷ and y > rain_threshold). The more extreme the true precipitation,
    the closer γ → 1 and the heavier the penalty.

    Args:
        train: Unused (API compatibility).
        gamma_shape (np.ndarray [H, W]): Pre-fitted Gamma shape parameters.
        gamma_scale (np.ndarray [H, W]): Pre-fitted Gamma scale parameters.
        rain_threshold: Dry/wet threshold (same units as data). Asymmetric term is
                        zeroed for pixels where y ≤ rain_threshold.
        reduce_mean: If True, average over all elements. If False, average per
                     sample then over the batch.

    Returns:
        A loss_fn(model, batch, cond, generator=None) callable.
    """
    if is_main_process():
        logger.info(
            " >> >> Doury ASYM loss initialised "
            "(rain_threshold=%.3f, H=%d, W=%d)",
            rain_threshold,
            gamma_shape.shape[0],
            gamma_shape.shape[1],
        )

    # Pre-convert gamma params to torch tensors (moved to device inside loss_fn)
    shape_t = torch.from_numpy(gamma_shape.astype(np.float32))
    scale_t = torch.from_numpy(gamma_scale.astype(np.float32))

    call_state = {"n": 0}

    def loss_fn(model, batch, cond, generator=None):
        call_state["n"] += 1

        x = torch.zeros_like(batch)
        t = torch.zeros(batch.shape[0], device=batch.device)
        pred = model(x, cond, t)

        target = batch   # physical units (mm/hr or mm/day)
        shape_dev = shape_t.to(batch.device)
        scale_dev = scale_t.to(batch.device)

        # ---------------------------------------------------------------- #
        # γ_{i,t} = Gamma CDF at target value, per grid point              #
        # Supports [B, H, W] and [B, C, H, W] layouts                     #
        # ---------------------------------------------------------------- #
        if target.ndim == 3:
            # [B, H, W]
            dist = torch.distributions.Gamma(
                concentration=shape_dev.unsqueeze(0),  # [1, H, W]
                rate=1.0 / scale_dev.unsqueeze(0),
            )
            gamma_cdf = dist.cdf(target.clamp(min=1e-8))
        elif target.ndim == 4:
            # [B, C, H, W] – apply spatial params to each channel
            dist = torch.distributions.Gamma(
                concentration=shape_dev.unsqueeze(0).unsqueeze(0),   # [1,1,H,W]
                rate=1.0 / scale_dev.unsqueeze(0).unsqueeze(0),
            )
            gamma_cdf = dist.cdf(target.clamp(min=1e-8))
        else:
            raise ValueError(
                f"Unexpected tensor shape {target.shape}; expected [B,H,W] or [B,C,H,W]."
            )

        # ---------------------------------------------------------------- #
        # MAE term                                                          #
        # ---------------------------------------------------------------- #
        mae = torch.abs(target - pred)

        # ---------------------------------------------------------------- #
        # Asymmetric term: γ² · max(0, y - ŷ), zero on dry pixels          #
        # ---------------------------------------------------------------- #
        underest = torch.clamp(target - pred, min=0.0)
        wet_mask = (target > rain_threshold).float()
        asym_term = (gamma_cdf ** 2) * underest * wet_mask

        element_loss = mae + asym_term

        if reduce_mean:
            loss = element_loss.mean()
        else:
            spatial_dims = tuple(range(1, element_loss.ndim))
            loss = element_loss.mean(dim=spatial_dims).mean()

        if is_main_process() and (call_state["n"] % 50 == 0):
            logger.info(
                " >> >> Doury ASYM loss | step %d: mae=%.6e, asym=%.6e, total=%.6e",
                call_state["n"],
                mae.mean().item(),
                asym_term.mean().item(),
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

        elif (config.training.det_loss_type == 'ASYM'):

            logger.info("[[[[[[[[[[[[[[[[[[[[[[[ Doury ASYM ]]]]]]]]]]]]]]]]]]]]]]]")

            assert config.data.predictands.target_transform_keys[0] == "none", (
                f"ASYM loss requires physical-unit targets (target_transform_keys='none'). "
                f"Got {config.data.predictands.target_transform_keys}"
            )

            # ------------------------------------------------------------ #
            # Auto-construct gamma params path:                             #
            #   WORK_DIR/doury_gamma_params/dataset_name/train_stem/parameters.npz
            # ------------------------------------------------------------ #
            work_dir = os.getenv('WORK_DIR', os.getenv('DERIVED_DATA', '.'))
            dataset_name = config.data.dataset_name
            train_stem = Path(config.training.asym_train_filename).stem
            params_path = (
                Path(work_dir)
                / 'doury_gamma_params'
                / dataset_name
                / train_stem
                / 'parameters.npz'
            )

            if is_main_process():
                logger.info("[ASYM] Gamma params path: %s", params_path)

            # ------------------------------------------------------------ #
            # Cache check: load if exists, otherwise fit and save           #
            # ------------------------------------------------------------ #
            if params_path.exists():
                if is_main_process():
                    logger.info(
                        "[ASYM] Pre-computed gamma params found — skipping fitting."
                    )
                    print(f"[ASYM] Loading gamma params from {params_path}")
                npz = np.load(params_path)
                gamma_shape = npz['shape']
                gamma_scale = npz['scale']
            else:
                if is_main_process():
                    logger.info(
                        "[ASYM] Gamma params not found at %s — fitting now.", params_path
                    )
                    print(f"[ASYM] Gamma params not found. Fitting from training data …")

                # Open training zarr and extract precipitation DataArray
                derived_data = os.getenv('DERIVED_DATA', '.')
                train_zarr_path = (
                    Path(derived_data)
                    / dataset_name
                    / config.training.asym_train_filename
                )
                try:
                    import xarray as xr
                    ds = xr.open_zarr(str(train_zarr_path), consolidated=True, chunks='auto')
                    precip_da = ds[config.data.predictands.variables[0]]
                except Exception as e:
                    raise RuntimeError(
                        f"[ASYM] Could not open training data at {train_zarr_path} "
                        f"to fit gamma params: {e}"
                    )

                gamma_shape, gamma_scale = fit_gamma_params_per_gridpoint(
                    precip_da,
                    rain_threshold=config.training.asym_rain_threshold,
                    lat_chunk=config.training.asym_lat_chunk,
                    lon_chunk=config.training.asym_lon_chunk,
                )

                # Save parameters
                params_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(params_path, shape=gamma_shape, scale=gamma_scale)
                if is_main_process():
                    logger.info("[ASYM] Gamma params saved to %s", params_path)
                    print(f"[ASYM] Gamma params saved to {params_path}")

            loss_fn = get_deterministic_loss_fn_asym_doury(
                train,
                gamma_shape=gamma_shape,
                gamma_scale=gamma_scale,
                rain_threshold=config.training.asym_rain_threshold,
                reduce_mean=config.training.reduce_mean,
            )

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