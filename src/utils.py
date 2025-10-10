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
# to allow for conditional modelling.

"""All functions and modules related to model definition.
"""
import sys
sys.dont_write_bytecode = True
import torch
print(" >> >> INSIDE utils")
from . import sde_lib
print(" >> >> INSIDE utils")
import numpy as np
print(" >> >> INSIDE utils")
import os
import logging
from pytorch_lightning.callbacks import ProgressBar, TQDMProgressBar, ModelCheckpoint
from pathlib import Path
import numpy as np
import torch.distributed as dist
import xarray as xr
from typing import Any
import os
import json
import ml_collections
import torch

logger = logging.getLogger()

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False
#====================================================================
_MODELS = {}
#====================================================================
def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    
    return score_model


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
        config: A ConfigDict object parsed from the config file
    Returns:
        sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, cond, labels):
        """Compute the output of the score-based model.

        Args:
        x: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
        A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model.forward(x, cond, labels)
        else:
            model.train()
            return model.forward(x, cond, labels)

    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
        A score function.
    """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, cond, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, cond, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, cond, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, cond, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state, False
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["location_params"].load_state_dict(loaded_state["location_params"])
        state["step"] = loaded_state["step"]
        state["epoch"] = loaded_state["epoch"]
        logger.info("    - -  -   -    -     -      -      -     -    -   -  - -")
        logger.info(
            f" >> Checkpoint found at {ckpt_dir}. "
            f" >> Returned the state from {state['epoch']}/{state['step']}"
        )
        logger.info("    - -  -   -    -     -      -      -     -    -   -  - -")
        return state, True


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
        "epoch": state["epoch"],
        "location_params": state["location_params"].state_dict(),
    }
    torch.save(saved_state, ckpt_dir)


def param_count(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def model_size(model):
    """Compute size in memory of model in MB."""
    param_size = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in model.buffers()
    )

    return (param_size + buffer_size) / 1024**2


class LossOnlyProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False
    
    def get_metrics(self, trainer, pl_module):
        # Get the default metrics
        metrics = super().get_metrics(trainer, pl_module)
        # Filter to only show train_loss and val_loss
        return {
            k: v for k, v in metrics.items()
            if k in ("train_loss", "val_loss")
        }


def setup_checkpoint(config, workdir):
    if ((config.experiment_name == '') or (config.experiment_name == None)):
        dirpath = os.path.join(workdir, 'checkpoints', config.data.dataset_name)
    else:
        dirpath = os.path.join(workdir, 'checkpoints', config.data.dataset_name, config.experiment_name)
    
    os.makedirs(dirpath, exist_ok=True)
    
    logger.info(" >> INSIDE utils.setup_checkpoint: %s", str(dirpath))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=config.model.name+'-'+config.training.sde+"-{epoch}",
        save_top_k=-1,
        every_n_epochs=config.training.snapshot_freq,
        save_last=True,
        save_weights_only=False,
    )

    return checkpoint_callback, dirpath


def check_saved_checkpoint(dirpath):
    ckpt_path = os.path.join(dirpath, "last.ckpt")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        last_epoch = checkpoint.get("epoch", "Not found")
        logger.info("    - -  -   -    -     -      -      -     -    -   -  - -")
        logger.info(" >> RESUMING TRAINING FROM EPOCH: %s %s", str(last_epoch), str(ckpt_path))
        logger.info("    - -  -   -    -     -      -      -     -    -   -  - -")
    else:
        logger.info("    - -  -   -    -     -      -      -     -    -   -  - -")
        logger.info(" >> NO CHECKPOINT FOUND AT %s TRAINING FROM SCRATCH", str(ckpt_path))
        logger.info("    - -  -   -    -     -      -      -     -    -   -  - -")
        ckpt_path = None
    
    return ckpt_path


def samples_path(
    workdir: str,
    checkpoint: str,
    dataset: str,
    filename: str,
    experiment_name = None) -> Path:
    filename = filename.split('.')[0] # Remove .blahblah from end
    checkpoint = checkpoint.split('.')[0]
    
    if ((experiment_name == '') or (type(experiment_name) == None)):
        return Path(workdir, "samples", dataset, filename, checkpoint)
    else:
        return Path(workdir, "samples", dataset, filename, checkpoint, experiment_name)


def load_sampling_config(workdir, dataset, train_config, experiment_name=None):
    if ((experiment_name == None) or (experiment_name == '')):
        if ((train_config.experiment_name == None) or (train_config.experiment_name == '')):
            config = load_config(os.path.join(workdir, 'checkpoints', dataset, 'config.yml'))
            logger.info(" >> >> INSIDE utils: sampling config: %s", os.path.join(workdir, 'checkpoints', dataset, 'config.yml'))
        else:
            config = load_config(os.path.join(workdir, 'checkpoints', dataset, train_config.experiment_name, 'config.yml'))
            logger.info(" >> >> INSIDE utils: sampling config: %s", os.path.join(workdir, 'checkpoints', dataset, train_config.experiment_name, 'config.yml'))
    else:
        config = load_config(os.path.join(workdir, 'checkpoints', dataset, experiment_name, 'config.yml'))
        logger.info(" >> >> INSIDE utils: sampling config: %s", os.path.join(workdir, 'checkpoints', dataset, experiment_name, 'config.yml'))

    return config


def make_predictions_filename(directory, config, prefix="predictions"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    num_scales = str(config.sampling.num_scales)

    existing_files = list(directory.glob(f"{prefix}_num_scales={num_scales}_sample*.nc"))
    next_index = len(existing_files)

    print(" >> >> INSIDE make_predictions_filename next_index,", next_index, ", ", len(list(directory.glob("*.nc"))))
    
    new_filename = prefix+'_num_scales='+str(num_scales)+'_sample'+str(next_index)+'.nc'
    return os.path.join(directory, new_filename)


def np_samples_to_xr(np_samples, target_transform, target_vars, coords, cf_data_vars):
    """
    Convert samples from a model in numpy format to an xarray Dataset, including inverting any transformation applied to the target variables before modelling.
    """
    coords = {**dict(coords)}

    pred_dims = ['time', 'lat', 'lon']

    for var_name in list(target_vars):
        ds = cf_data_vars[var_name]
        sub_ds = ds.sel(time=coords['time'])

    data_vars = {**cf_data_vars}
    for var_idx, var in enumerate(target_vars):
        # add ensemble member axis to np samples and get just values for current variable
        np_var_pred = np.squeeze(np_samples[:, var_idx, ...])
        pred_attrs = {
            "standard_name": var,
            "units": "mm/hr"
        }
        pred_var = (pred_dims, np_var_pred, pred_attrs)

        data_vars.update(
            {
                var: pred_var,  # don't rename pred var until after inverting target transform
                #var.replace("target_", "raw_pred_"): raw_pred_var,
            }
        )
    
    samples_ds = target_transform.invert(
        xr.Dataset(data_vars=data_vars, coords=coords, attrs={})
    )
    for var_idx, var in enumerate(target_vars):
        pred_attrs = {
            "grid_mapping": "rotated_latitude_longitude",
            #"standard_name": var.replace("target_", "pred_"),
            "standard_name": var,
            #"units": "kg m-2 s-1",
            "units": "mm/hr"
        }
        samples_ds[var].assign_attrs(pred_attrs)
    return samples_ds


def get_xarray_info(ds):
    lat_variants = ['lat', 'latitude', 'LAT', 'Latitude']
    lon_variants = ['lon', 'longitude', 'LON', 'Longitude']
    time_variants = ['time', 'TIME', 'Time']

    # Helper to find matching dimension or coordinate name
    def find_dim_name(ds, variants):
        for name in variants:
            if name in ds.dims or name in ds.coords:
                return name
        raise ValueError(f"No matching dimension found for variants: {variants}")

    # Find actual names
    lat_name = find_dim_name(ds, lat_variants)
    lon_name = find_dim_name(ds, lon_variants)
    time_name = find_dim_name(ds, time_variants)

    # Extract values and units
    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values
    time_values = ds[time_name].values

    lat_units = ds[lat_name].attrs.get('units', 'unknown')
    lon_units = ds[lon_name].attrs.get('units', 'unknown')
    time_units = ds[time_name].attrs.get('units', 'unknown')

    return {lat_name: (lat_values, lat_units), lon_name: (lon_values, lon_units), time_name: (time_values, time_units)}


def input_to_list(var):
    # If input is a string, wrap in a list
    if isinstance(var, str):
        return [var]
    # If input is a list or tuple, flatten one level and ensure all elements are strings
    elif isinstance(var, (list, tuple)):
        # Flatten if it's a nested list of strings
        flat = []
        for item in var:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, (list, tuple)):
                # Only flatten one level deep
                flat.extend(item)
            else:
                raise ValueError(f"Unsupported type in list: {type(item)}")
        return flat
    else:
        raise ValueError(f"Unsupported input type: {type(var)}")


def _to_primitive(obj: Any) -> Any:
    """
    Convert obj recursively into JSON/YAML-serializable primitives.
    Special-case: torch.device -> device.type (e.g. "cuda", "cpu").
    """
    # ml_collections.ConfigDict -> dict
    if isinstance(obj, ml_collections.ConfigDict):
        return {k: _to_primitive(v) for k, v in obj.items()}

    # dict
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}

    # tuple -> list
    if isinstance(obj, tuple):
        return [_to_primitive(v) for v in obj]

    # list
    if isinstance(obj, list):
        return [_to_primitive(v) for v in obj]

    # torch.device -> simple string of device type
    if isinstance(obj, torch.device):
        # per your request: store only 'cuda', 'cpu', etc.
        return obj.type

    # numpy scalars / arrays
    if _HAS_NUMPY:
        if isinstance(obj, _np.generic):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        if isinstance(obj, _np.ndarray):
            try:
                return obj.tolist()
            except Exception:
                return str(obj)

    # primitives
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Fallback: stringify unknown objects
    return str(obj)


def save_config(config: ml_collections.ConfigDict, dest_path: str) -> None:
    """
    Save an ml_collections.ConfigDict to JSON or YAML.
    - dest_path extension .yml/.yaml -> YAML (PyYAML required)
    - otherwise -> JSON
    """
    if not isinstance(config, ml_collections.ConfigDict):
        raise TypeError("`config` must be an ml_collections.ConfigDict")

    prim = _to_primitive(config)

    _, ext = os.path.splitext(dest_path)
    ext = (ext or ".json").lower()

    if ext in (".yml", ".yaml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to write YAML. Install with `pip install pyyaml`.")
        with open(dest_path, "w") as f:
            yaml.safe_dump(prim, f, sort_keys=False)
    else:
        # default to JSON
        with open(dest_path, "w") as f:
            json.dump(prim, f, indent=2, sort_keys=False)


def _dict_to_config(raw: Any) -> Any:
    """
    Convert a nested dict/list/primitive structure (from JSON/YAML) into
    ml_collections.ConfigDict (for dicts) and leave lists/primitives as-is.
    Note: device strings remain strings (no torch.device conversion).
    """
    if isinstance(raw, dict):
        cd = ml_collections.ConfigDict()
        for k, v in raw.items():
            cd[k] = _dict_to_config(v)
        return cd

    if isinstance(raw, list):
        return [_dict_to_config(x) for x in raw]

    # primitives (int/float/str/bool/None) -> return as is
    return raw


def load_config(path: str) -> ml_collections.ConfigDict:
    """
    Load a JSON/YAML config saved by save_config and return an ml_collections.ConfigDict.
    Device values will be plain strings ('cuda', 'cpu', etc.) if they were saved as torch.device.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")

    _, ext = os.path.splitext(path)
    ext = (ext or ".json").lower()

    if ext in (".yml", ".yaml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to read YAML. Install with `pip install pyyaml`.")
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            raw = json.load(f)

    if raw is None:
        return ml_collections.ConfigDict()

    cfg = _dict_to_config(raw)
    # ensure we return a ConfigDict
    if isinstance(cfg, ml_collections.ConfigDict):
        return cfg
    else:
        out = ml_collections.ConfigDict()
        out["value"] = cfg
        return out


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0