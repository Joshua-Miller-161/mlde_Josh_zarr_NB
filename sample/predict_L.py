"""Generate samples"""
import sys
sys.dont_write_bytecode = True
import os
os.environ["RICH_TRACEBACK"] = "false"
os.environ["RICH_NO_COLOR"] = "1"
os.environ["RICH_NO_STYLE"] = "1"
os.environ["TERM"] = "dumb"
from collections import defaultdict
import itertools
import os
from pathlib import Path
from codetiming import Timer
from dotenv import load_dotenv
from ml_collections import config_dict
import torch
import typer
from tqdm import tqdm
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr
import yaml
import time
import numpy as np

sys.path.append(os.getcwd())
from configs.subvpsde.ukcp_local_pr_1em_cncsnpp_continuous import get_config
from src.utils import make_predictions_filename, get_xarray_info, samples_path

#from src.data_scripts.collate_np.data_module import LightningDataModule
from src.data_scripts.collate_np_per_var.data_module import LightningDataModule

from src.optimizers import get_optimizer
from src.ema import ExponentialMovingAverage
from src.lightningModuleEMA import ScoreModelLightningModule
from src.cncsnpp import cNCSNpp  # noqa: F401
from sampling import get_sampling_fn
from src.sde_lib import VESDE, VPSDE, subVPSDE
from src.data_scripts.data_utils import datafile_path
#====================================================================
load_dotenv()  # take environment variables from .env.

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()

def load_config(config_path):
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = config_dict.ConfigDict(yaml.unsafe_load(f))

    return config

app = typer.Typer(pretty_exceptions_enable=False, rich_markup_mode=None)

def load_model(config, ckpt_path):
    deterministic = "deterministic" in config and config.deterministic
    if deterministic:
        sde = None
        sampling_eps = 0
    else:
        if config.training.sde == "vesde":
            sde = VESDE(
                sigma_min=config.model.sigma_min,
                sigma_max=config.model.sigma_max,
                N=config.sampling.num_scales,
            )
            sampling_eps = 1e-5
        elif config.training.sde == "vpsde":
            sde = VPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.sampling.num_scales,
            )
            sampling_eps = 1e-3
        elif config.training.sde == "subvpsde":
            sde = subVPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.sampling.num_scales,
            )
            sampling_eps = 1e-3
        else:
            raise RuntimeError(f"Unknown SDE {config.training.sde}")


    print(" >> INSIDE predict_v2.load_model config.model.num_scales =", config.model.num_scales, ", config.sampling.num_scales", config.sampling.num_scales)

    retreived_model = ScoreModelLightningModule.load_from_checkpoint(ckpt_path, config=config)
    retreived_model.to(config.device)
    retreived_model.ema.copy_to(retreived_model.model.parameters())

    # Sampling
    if not 'per_var' in LightningDataModule.__module__:
        from src.data_scripts.data_utils import get_variables
        input_variables, target_vars = get_variables(config.data.dataset_name)
    else:
        from src.data_scripts.collate_np_per_var import get_variables
        input_variables, target_vars = get_variables(config)

    num_output_channels = len(target_vars)
    sampling_shape = (
        config.eval.batch_size,
        num_output_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    return retreived_model, sampling_fn, target_vars


def generate_np_sample_batch(sampling_fn, score_model, config, cond_batch):
    cond_batch = cond_batch.to(config.device)

    samples = sampling_fn(score_model, cond_batch)[0]

    # extract numpy array
    samples = samples.cpu().numpy()

    print(" >> INSIDE generate_np_sample_batch samples.shape", samples.shape)

    return samples


def np_samples_to_xr(np_samples, target_transform, target_var, ref_ds, time_batch):
    """
    Convert samples from a model in numpy format to an xarray Dataset, including inverting any transformation applied to the target variables before modelling.
    """

    inverted_data = target_transform.invert(np_samples)

    print(" >> >> INSIDE np_samples_to_xr inverted_data ", type(inverted_data), inverted_data.shape)
    
    pred_da = xr.DataArray(
        data = np.squeeze(inverted_data),
        coords = {'time': time_batch,
                  'lat': ref_ds['lat'],
                  'lon': ref_ds['lon']},
        dims = tuple(ref_ds.dims),
        name = target_var,
        attrs = ref_ds.attrs.copy()
    )

    pred_da.attrs["standard_name"] = target_var

    logger.info("____________________________________________________________________________________")
    logger.info(f" << << INSIDE np_samples_to_xr dims type: {type(ref_ds.dims)}, tup: {tuple(ref_ds.dims)}, name: {target_var}")
    #logger.info(f" >> >> INSIDE np_samples_to_xr pred_da {pred_da}")
    logger.info("____________________________________________________________________________________")
    return pred_da


def sample(sampling_fn, score_model, config, eval_dl, target_transform, target_var, reference_file):
    # cf_data_vars = {key: eval_dl.dataset.ds.data_vars[key]
    #                 for key in list(eval_dl.dataset.ds.data_vars)}

    x_init = next(iter(eval_dl))
    print(" >> >> INSIDE predict_v2.sample")
    print("________________________________________________________________________________")
    print(" >> x_init", type(x_init), len(x_init), type(eval_dl))
    print(" >> x_init[0]", x_init[0].shape)
    print(" >> x_init[1]", x_init[1].shape)
    print(" >> x_init[2]", x_init[2].shape, len(x_init[2].shape))
    logger.info("________________________________________________________________________________")
    logger.info(f" >> >> INSIDE predict_L.sample target_transform {type(target_transform)} {target_transform}")
    logger.info("________________________________________________________________________________")
    
    ref_ds = xr.open_zarr(reference_file)[target_var]

    target_var = target_var[0]

    xr_sample_batches = []
    with logging_redirect_tqdm():
        with tqdm(
            total=len(eval_dl.dataset),
            desc=f"Sampling",
            unit=" timesteps",
        ) as pbar:
            for cond_batch, _, time_batch in eval_dl:
                # append any location-specific parameters
                #cond_batch = location_params(cond_batch)
                print(" >> >> INSIDE sample time_batch =", time_batch.shape)#, time_batch)
                logger.info(f" >> >> INSIDE sample time_batch = {time_batch.shape}")# {time_batch}")
                #coords = eval_dl.dataset.ds.sel(time=time_batch).coords

                print(" >> >> INSIDE sample cond_batch", cond_batch.shape)
                logger.info(f" >> >> INSIDE sample cond_batch = {cond_batch.shape}")

                np_sample_batch = generate_np_sample_batch(
                    sampling_fn,
                    score_model,
                    config,
                    cond_batch
                )

                xr_sample_batch = np_samples_to_xr(
                    np_sample_batch,
                    target_transform[target_var],
                    target_var,
                    ref_ds,
                    time_batch
                )

                print("_________________ INSIDE SAMPLE _________________")
                print(xr_sample_batch.sizes)
                print("_________________________________________________")

                logger.info("_________________ INSIDE SAMPLE _________________")
                logger.info(f" >> sizes: {xr_sample_batch.sizes} xr_sample_batch_coords: {xr_sample_batch.coords}")
                logger.info("_________________________________________________")

                xr_sample_batches.append(xr_sample_batch)

                pbar.update(cond_batch.shape[0])

    logger.info("==============================================================")
    logger.info("==============================================================")
    logger.info(' >> >> INSIDE sample type(xr_sample_batches) %s len %d', type(xr_sample_batches), len(xr_sample_batches))
    logger.info("______________________________________________________________")
    for da in xr_sample_batches:
        logger.info(f" >> sizes: {da.sizes}, type: {type(da)}, name: {da.name}, coords: {da.coords}, type(da.coords): {type(da.coords)}, dims: {da.dims}, type(da.dims) {type(da.dims)}")
        logger.info("______________________________________________________________")
    logger.info("==============================================================")
    logger.info("==============================================================")

    ds = xr.concat(xr_sample_batches, dim='time')
    ds = ds.sortby('time')
    #ds = ds.sel(time=~ds.time.duplicated())
    logger.info("==============================================================")
    logger.info("==============================================================")
    logger.info(f" >> sizes:{ds.sizes}, name: {ds.name}, coords: {ds.coords}, dims: {ds.dims}")
    return ds


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def main(
    workdir: Path,
    dataset: str = typer.Option(...),
    filename: str = "val",
    checkpoint: str = typer.Option(...),
    batch_size: int = None,
    num_samples: int = 3,
    input_transform_dataset: str = None,
    input_transform_key: str = None,
    num_scales: int = None,
    experiment_name: str = None
):
    print(" >> INSIDE predict_L.main input_transform_dataset", input_transform_dataset)
    config = get_config()

    with config.unlocked():
        if num_scales is not None:
            config.sampling.num_scales = num_scales
    
        if batch_size is not None:
            config.eval.batch_size = batch_size
        
        if (dataset is not None) or (dataset != ''):
            config.data.dataset_name = dataset
        
        if input_transform_dataset is not None:
            config.data.input_transform_dataset = dataset
        else:
            config.data.input_transform_dataset = input_transform_dataset

        if input_transform_key is not None:
            config.data.input_transform_key = input_transform_key

    print(" >> INSIDE predict_L config.model.num_scales", config.model.num_scales, ", config.sampling.num_scales", config.sampling.num_scales)
    print(" >> INSIDE predict_L.main config.data.input_transform_dataset", config.data.input_transform_dataset)

    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=checkpoint,
        dataset=dataset,
        input_xfm=f"{config.data.input_transform_dataset}-{config.data.input_transform_key}",
        filename=filename,
        experiment_name=experiment_name
    )

    os.makedirs(output_dirpath, exist_ok=True)

    sampling_config_path = os.path.join(output_dirpath, "config.yml")
    with open(sampling_config_path, "w") as f:
        f.write(config.to_yaml())

    if not 'per_var' in LightningDataModule.__module__:
        data_module = LightningDataModule(
            active_dataset_name=config.data.dataset_name,
            model_src_dataset_name=config.data.dataset_name,
            input_transform_dataset_name=config.data.dataset_name,
            input_transform_key=config.data.input_transform_key,
            target_transform_key=config.data.target_transform_key, # Orig, target_transform_keys = target_xfm_keys
            transform_dir=os.path.join(workdir, 'transforms'),
            batch_size=config.eval.batch_size,
            filename=filename,
            include_time_inputs=False,
            evaluation=False,
            shuffle=False
        )
    else:
        data_module = LightningDataModule(
            config=config,
            active_dataset_name=config.data.dataset_name,
            model_src_dataset_name=config.data.dataset_name,
            input_transform_dataset_name=config.data.dataset_name,
            transform_dir=os.path.join(workdir, 'transforms'),
            batch_size=config.eval.batch_size, #config.training.batch_size,
            filename=filename,
            include_time_inputs=False,
            evaluation=False,
            shuffle=False
        )
    
    start_time = time.time()
    data_module.setup("test")
    end_time = time.time()
    print(f" <> INSIDE predict_L data_module.setup('test') {end_time-start_time:.3f} seconds")

    eval_dl = data_module.test_dataloader()
    with config.unlocked():
        x_init = next(iter(eval_dl))
        config.data.image_size = x_init[0].shape[-1]

    ckpt_filename = os.path.join(workdir, "checkpoints", config.data.dataset_name+'_'+config.experiment_name, checkpoint)
    logger.info(f"Loading model from {ckpt_filename}")
    score_model, sampling_fn, target_var = load_model(config, ckpt_filename)

    print(" >> INSIDE main score_model", type(score_model), dir(score_model))
    
    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")

        start_time = time.time()
        xr_samples = sample(sampling_fn,
                            score_model,
                            config,
                            eval_dl,
                            data_module.test_target_transforms,
                            target_var,
                            datafile_path(config.data.dataset_name, filename))
        end_time = time.time()
        print(f" <> INSIDE predict_L.py sample {end_time-start_time:.3f} seconds")

        output_filepath = make_predictions_filename(output_dirpath, config)

        logger.info(f"Saving samples to {output_filepath}...")
        
        start_time = time.time()
        xr_samples.to_netcdf(output_filepath)
        end_time = time.time()
        print(f" <> INSIDE predict_L.py xr_samples.to_netcdf {end_time-start_time:.3f} seconds")

if __name__ == "__main__":
    app()