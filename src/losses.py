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
    precip_data,
    precip_var='precipitation',
    rain_threshold=1.0,
    lat_chunk=16,
    lon_chunk=16,
    min_rainy_days=10,
):
    """Fit Gamma(shape, scale) at each grid point on rainy-day values.

    Designed for large (tens of GB) dask-backed precipitation DataArrays.
    Data is rechunked to (time=-1, lat_chunk, lon_chunk) tiles for fitting;
    the original DataArray passed in is never mutated.

    Args:
        precip_data: One of:
            - str/Path    : path to a zarr or NetCDF file
            - xr.DataArray: dask-backed or in-memory array (time, lat, lon)
            - xr.Dataset  : dataset; ``precip_var`` is extracted
            - np.ndarray  : shape (T, H, W)
        precip_var: Variable name if ``precip_data`` is a file path or Dataset.
        rain_threshold: Wet-day threshold (same units as data).
        lat_chunk: Spatial tile size in the latitude dimension.
        lon_chunk: Spatial tile size in the longitude dimension.
        min_rainy_days: Minimum rainy samples required for a proper fit;
            grid points with fewer samples fall back to Exponential(shape=1).

    Returns:
        shape_map: np.ndarray [H, W] of Gamma shape parameters.
        scale_map: np.ndarray [H, W] of Gamma scale parameters.
    """
    import xarray as xr
    import scipy.stats
    from tqdm import tqdm

    # --- Load / prepare DataArray ---
    if isinstance(precip_data, (str, Path)):
        p = Path(precip_data)
        if p.suffix == '.zarr' or p.is_dir():
            ds = xr.open_zarr(str(p))
        else:
            ds = xr.open_dataset(str(p), chunks='auto')
        da = ds[precip_var]
    elif isinstance(precip_data, xr.Dataset):
        da = precip_data[precip_var]
    elif isinstance(precip_data, xr.DataArray):
        da = precip_data
    elif isinstance(precip_data, np.ndarray):
        da = xr.DataArray(precip_data, dims=['time', 'lat', 'lon'])
    else:
        raise TypeError(f"Unsupported type for precip_data: {type(precip_data)}")

    dims     = list(da.dims)
    time_dim = dims[0]
    lat_dim  = dims[1]
    lon_dim  = dims[2]
    H = da.sizes[lat_dim]
    W = da.sizes[lon_dim]

    # --- Spin up LocalCluster for parallel tile IO ---
    client  = None
    cluster = None
    try:
        from dask.distributed import LocalCluster, Client
        n_workers = max(1, os.cpu_count() or 1)
        print(f"[fit_gamma] Starting LocalCluster with {n_workers} worker(s)...")
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, memory_limit='4GB')
        client  = Client(cluster)
        print(f"[fit_gamma] Dashboard: {client.dashboard_link}")
    except ImportError:
        print("[fit_gamma] dask.distributed not available; using synchronous scheduler")

    # --- Rechunk for fitting: full time axis, small spatial tiles ---
    da_fit = da.chunk({time_dim: -1, lat_dim: lat_chunk, lon_dim: lon_chunk})

    # --- Allocate output ---
    shape_map    = np.ones((H, W), dtype=np.float32)
    scale_map    = np.ones((H, W), dtype=np.float32)
    fallback_cnt = 0

    n_lat_tiles = (H + lat_chunk - 1) // lat_chunk
    n_lon_tiles = (W + lon_chunk - 1) // lon_chunk
    n_tiles     = n_lat_tiles * n_lon_tiles

    with tqdm(total=n_tiles, desc="Tiles loaded", unit="tile") as tile_pbar:
        with tqdm(total=H * W, desc="Gridpoints fitted", unit="pt") as pt_pbar:
            for i0 in range(0, H, lat_chunk):
                i1 = min(i0 + lat_chunk, H)
                for j0 in range(0, W, lon_chunk):
                    j1 = min(j0 + lon_chunk, W)

                    # Compute tile (triggers dask IO, parallelised by LocalCluster)
                    tile = da_fit.isel(
                        {lat_dim: slice(i0, i1), lon_dim: slice(j0, j1)}
                    ).values  # [T, tile_h, tile_w]

                    tile_h = i1 - i0
                    tile_w = j1 - j0

                    with tqdm(
                        total=tile_h * tile_w,
                        desc=f"Rows {i0}-{i1}",
                        leave=False,
                        unit="pt",
                    ) as inner_pbar:
                        for ii in range(tile_h):
                            for jj in range(tile_w):
                                series = tile[:, ii, jj].astype(np.float64)
                                rainy  = series[series > rain_threshold]
                                gi, gj = i0 + ii, j0 + jj

                                if len(rainy) >= min_rainy_days:
                                    try:
                                        shp, _, scl = scipy.stats.gamma.fit(rainy, floc=0)
                                        shape_map[gi, gj] = max(float(shp), 1e-6)
                                        scale_map[gi, gj] = max(float(scl), 1e-6)
                                    except Exception:
                                        shape_map[gi, gj] = 1.0
                                        scale_map[gi, gj] = max(float(np.mean(rainy)), 1e-6)
                                        fallback_cnt += 1
                                else:
                                    shape_map[gi, gj] = 1.0
                                    scale_map[gi, gj] = max(
                                        float(np.mean(rainy)) if len(rainy) > 0 else 1.0,
                                        1e-6,
                                    )
                                    fallback_cnt += 1

                                inner_pbar.update(1)
                                pt_pbar.update(1)

                    tile_pbar.update(1)

    if client is not None:
        client.close()
        cluster.close()

    total = H * W
    print(f"\n[fit_gamma] Complete — {total} gridpoints | "
          f"fitted: {total - fallback_cnt} | fallbacks: {fallback_cnt}")
    print(f"[fit_gamma] shape: min={shape_map.min():.4f} max={shape_map.max():.4f} "
          f"mean={shape_map.mean():.4f}")
    print(f"[fit_gamma] scale: min={scale_map.min():.4f} max={scale_map.max():.4f} "
          f"mean={scale_map.mean():.4f}")

    return shape_map, scale_map
#====================================================================
def get_deterministic_loss_fn_asym_doury(
    train,
    gamma_shape,
    gamma_scale,
    rain_threshold=1.0,
    reduce_mean=True,
):
    """Doury et al. (2024) asymmetric precipitation loss (Eq. 4).

    L(y, ŷ) = mean( |y - ŷ|  +  γ²_{i,t} · max(0, y_{i,t} - ŷ_{i,t}) )

    where γ_{i,t} = G_i(y_{i,t}) is the per-gridpoint Gamma CDF evaluated
    at the true precipitation value.  The asymmetric term fires only when
    the model underestimates a rainy pixel, with a penalty proportional to
    the rarity of that rainfall intensity.

    The model must be trained in physical units (target_transform_keys='none').

    Args:
        train: Unused; kept for API compatibility.
        gamma_shape: [H, W] numpy array of fitted Gamma shape parameters.
        gamma_scale: [H, W] numpy array of fitted Gamma scale parameters.
        rain_threshold: Wet-day threshold (same units as data).
            Dry pixels (y ≤ threshold) have the asymmetric term zeroed.
        reduce_mean: If True, average loss over all elements.

    Returns:
        A loss_fn(model, batch, cond, generator=None) callable.
    """
    gamma_shape_np = np.asarray(gamma_shape, dtype=np.float32)
    gamma_scale_np = np.asarray(gamma_scale, dtype=np.float32)

    if is_main_process():
        logger.info(
            "[asym_doury] loss initialised | rain_threshold=%.3f | "
            "shape mean=%.4f | scale mean=%.4f",
            rain_threshold,
            gamma_shape_np.mean(),
            gamma_scale_np.mean(),
        )

    # Cache tensors per device to avoid re-creating every step
    _cache = {"device": None, "shape_t": None, "scale_t": None}
    call_state = {"n": 0}

    def loss_fn(model, batch, cond, generator=None):
        call_state["n"] += 1
        device = batch.device

        if _cache["device"] != device:
            _cache["shape_t"] = torch.tensor(gamma_shape_np, device=device)
            _cache["scale_t"] = torch.tensor(gamma_scale_np, device=device)
            _cache["device"]  = device

        shape_t = _cache["shape_t"]
        scale_t = _cache["scale_t"]

        # Deterministic model: x=0, t=0
        x    = torch.zeros_like(batch)
        t    = torch.zeros(batch.shape[0], device=device)
        pred = model(x, cond, t)

        target_phys = batch
        pred_phys   = pred

        # --- Per-gridpoint Gamma CDF (no gradient needed — applied to target) ---
        # CDF of Gamma(shape, scale) at y = gammainc(shape, y/scale)
        with torch.no_grad():
            target_safe = torch.clamp(target_phys, min=0.0)

            if target_phys.ndim == 4:          # [B, C, H, W]
                shape_exp = shape_t.unsqueeze(0).unsqueeze(0)
                scale_exp = scale_t.unsqueeze(0).unsqueeze(0)
            else:                              # [B, H, W]
                shape_exp = shape_t.unsqueeze(0)
                scale_exp = scale_t.unsqueeze(0)

            gamma_cdf = torch.special.gammainc(
                shape_exp,
                target_safe / scale_exp.clamp(min=1e-8),
            )
            # Zero asymmetric term on dry pixels
            rainy_mask = (target_phys > rain_threshold).float()
            gamma_cdf  = gamma_cdf * rainy_mask

        # --- MAE term ---
        mae = torch.abs(target_phys - pred_phys)

        # --- Asymmetric term: γ² · max(0, y - ŷ) ---
        underest = torch.clamp(target_phys - pred_phys, min=0.0)
        asym     = gamma_cdf ** 2 * underest

        total = mae + asym

        # --- Reduce ---
        if reduce_mean:
            loss     = torch.mean(total)
            mae_val  = torch.mean(mae)
            asym_val = torch.mean(asym)
        else:
            if total.ndim == 4:
                sdims = (1, 2, 3)
            elif total.ndim == 3:
                sdims = (1, 2)
            else:
                sdims = tuple(range(1, total.ndim))
            loss     = total.mean(dim=sdims).mean()
            mae_val  = mae.mean(dim=sdims).mean()
            asym_val = asym.mean(dim=sdims).mean()

        if is_main_process() and (call_state["n"] % 50 == 0):
            logger.info(
                " >> >> [asym_doury] step %d: mae=%.6e asym=%.6e total=%.6e",
                call_state["n"],
                mae_val.item(),
                asym_val.item(),
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

        elif config.training.det_loss_type == 'ASYM':

            logger.info("[[[[[[[[[[[[[[[[[[[[[[[ ASYM (Doury et al. 2024) ]]]]]]]]]]]]]]]]]]]]]]]")

            assert config.data.predictands.target_transform_keys[0] == "none", (
                f"ASYM loss requires physical-unit targets (target_transform_keys must be 'none'). "
                f"Got {config.data.predictands.target_transform_keys}"
            )

            # --- Determine training filename ---
            # If asym_train_filename is not set, fall back to FLAGS.filename
            # (the same zarr used for ML training), so no extra config is needed.
            asym_train_filename = getattr(config.training, 'asym_train_filename', '')
            if not asym_train_filename:
                from absl import flags as _absl_flags
                asym_train_filename = _absl_flags.FLAGS.filename
                logger.info(
                    "[get_loss/ASYM] asym_train_filename not set; "
                    "using ML training file: %s", asym_train_filename
                )

            # --- Auto-construct gamma params path ---
            # $WORK_DIR/doury_gamma_params/<dataset_name>/<train_stem>/parameters.npz
            work_dir    = os.getenv('WORK_DIR', os.getenv('DERIVED_DATA', '.'))
            train_stem  = Path(asym_train_filename).stem
            params_path = (
                Path(work_dir)
                / 'doury_gamma_params'
                / config.data.dataset_name
                / train_stem
                / 'parameters.npz'
            )
            logger.info("[get_loss/ASYM] gamma params path: %s", params_path)

            # --- Cache check: skip fitting if params already exist ---
            if params_path.exists():
                logger.info("[get_loss/ASYM] Loading cached gamma params from %s", params_path)
                npz         = np.load(params_path)
                gamma_shape = npz['shape']
                gamma_scale = npz['scale']
            else:
                logger.info("[get_loss/ASYM] Fitting gamma params from %s", asym_train_filename)
                import xarray as xr
                derived_data = os.getenv('DERIVED_DATA', '.')
                zarr_path    = os.path.join(derived_data, config.data.dataset_name, asym_train_filename)
                ds           = xr.open_zarr(zarr_path)
                precip_var   = config.data.predictands.variables[0]
                precip_da    = ds[precip_var]

                gamma_shape, gamma_scale = fit_gamma_params_per_gridpoint(
                    precip_da,
                    rain_threshold=config.training.asym_rain_threshold,
                    lat_chunk=config.training.asym_lat_chunk,
                    lon_chunk=config.training.asym_lon_chunk,
                )
                params_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(params_path, shape=gamma_shape, scale=gamma_scale)
                logger.info("[get_loss/ASYM] Saved gamma params to %s", params_path)

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