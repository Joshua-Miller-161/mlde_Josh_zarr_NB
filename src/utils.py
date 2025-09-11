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

#from torchview import draw_graph

_MODELS = {}


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
        logging.info(
            f"Checkpoint found at {ckpt_dir}. "
            f"Returned the state from {state['epoch']}/{state['step']}"
        )
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
    
    dirpath = os.path.join(workdir, os.path.join('checkpoints', config.data.dataset_name))
    os.makedirs(dirpath, exist_ok=True)
    
    print(" >> INSIDE utils.setup_checkpoint: ", dirpath)
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
        print(f" >> RESUMING TRAINING FROM EPOCH: {last_epoch}")
        print(" >> >>", ckpt_path)
    else:
        print(" >> NO CHECKPOINT FOUND AT", ckpt_path, " TRAINING FROM SCRATCH")
        ckpt_path = None
    
    return ckpt_path


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

    for var_name in list(cf_data_vars.keys()):
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

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0



