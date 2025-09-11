import sys
sys.dont_write_bytecode = True
import os
import numpy as np
from torch.utils.data import default_collate
import xarray as xr
import cftime
from pathlib import Path
import yaml
import logging
import gc
from datetime import timedelta
from flufl.lock import Lock
import time
from torch.utils.data import DataLoader
import torch.distributed as dist
from datetime import datetime

logger = logging.getLogger()

from .original.dataset import DownscalingDataset
from ..ml_downscaling_emulator.mlde_josh_utils.transforms import build_input_transform, build_target_transform, _build_target_transform, save_transform, load_transform
#====================================================================
# ''' Handles printing and logging from multiple GPUs (from ChatGPT lol)'''

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
#====================================================================
def dataset_path(dataset: str, base_dir: str = None) -> Path:
    if base_dir is None:
        base_dir = os.getenv("DERIVED_DATA")

    print(" >> >> INSIDE dataset_path", Path(base_dir, dataset))
    
    return Path(base_dir, dataset)
#====================================================================
def datafile_path(dataset: str, filename: str, base_dir: str = None) -> Path:
    return dataset_path(dataset, base_dir=base_dir) / filename
#====================================================================
def dataset_config_path(dataset: str, base_dir: str = None) -> Path:

    print(" >> >> INSIDE dataset_config_path", dataset_path(dataset, base_dir=base_dir) / "ds-config.yml")

    return dataset_path(dataset, base_dir=base_dir) / "ds-config.yml"
#====================================================================
def dataset_config(dataset: str, base_dir: str = None) -> dict:
    with open(dataset_config_path(dataset, base_dir=base_dir), "r") as f:
        return yaml.safe_load(f)
#====================================================================
def open_zarr(dataset_name, filename):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if is_main_process():
        print(" >> >> INSIDE mlde_josh_utils.data.data_utils.py open_zarr,", datafile_path(dataset_name, filename))
    logger.info(" >> >> INSIDE mlde_josh_utils.data.data_utils.py open_zarr [Rank %d], %s", rank, str(datafile_path(dataset_name, filename)))
    return xr.open_zarr(datafile_path(dataset_name, filename), consolidated=True)
#====================================================================
def build_DataLoader(xr_data, model_src_dataset_name, batch_size, shuffle, include_time_inputs):
    # sampler = DistributedSampler(dataset) if self.trainer and self.trainer.world_size > 1 else None
    if is_main_process():
        print(" >> >> inside lightningDataModule._wrap_loader", type(xr_data))
    time_range = None
    if include_time_inputs:
        time_range = TIME_RANGE

    variables, target_variables = get_variables(model_src_dataset_name)

    xr_dataset = DownscalingDataset(xr_data,
                                    variables,
                                    target_variables,
                                    time_range)

    data_loader = DataLoader(xr_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=custom_collate)
        
    return data_loader
#====================================================================
def get_variables(dataset_name):
    print(" >> >> INSIDE get_variables dataset_name", dataset_name)
    ds_config = dataset_config(dataset_name)

    variables = ds_config["predictors"]["variables"]
    target_variables = list(
        #map(lambda v: f"target_{v}", ds_config["predictands"]["variables"])
        ds_config["predictands"]["variables"]
    )

    #print(" >> >> target_variables", target_variables)
    return variables, target_variables
#====================================================================
def _build_transform(
    filename,
    variables,
    active_dataset_name,
    model_src_dataset_name,
    transform_keys,
    builder,
):
    logging.info(f"Fitting transform")

    xfm = builder(variables, transform_keys)

    model_src_training_split = open_zarr(model_src_dataset_name, filename)
    print(" >> >> INSIDE mlde_josh_utils.data.data_utils.py _build_transform: model_src_training_split = load_raw_dataset(model_src_dataset_name, filename)")

    active_dataset_training_split = open_zarr(active_dataset_name, filename)
    print(" >> >> INSIDE mlde_josh_utils.data.data_utils.py _build_transform: active_dataset_training_split = load_raw_dataset(active_dataset_name, filename)") 
          
    xfm.fit(active_dataset_training_split, model_src_training_split)

    print(" >> >> INSIDE mlde_josh_utils.data.data_utils.py _build_transform: xfm.fit(active_dataset_training_split, model_src_training_split)") 
    
    model_src_training_split.close()
    del model_src_training_split
    active_dataset_training_split.close()
    del active_dataset_training_split
    gc.collect

    return xfm
#====================================================================
def _find_or_create_transforms(
    filename,
    active_dataset_name,
    model_src_dataset_name,
    transform_dir,
    input_transform_key,
    target_transform_keys,
    evaluation,
):
    variables, target_variables = get_variables(model_src_dataset_name)

    if transform_dir is None:
        input_transform = _build_transform(
            filename,
            variables,
            active_dataset_name,
            model_src_dataset_name,
            input_transform_key,
            build_input_transform,
        )

        if evaluation:
            raise RuntimeError("Target transform should only be fitted during training")
        target_transform = _build_transform(
            filename,
            target_variables,
            active_dataset_name,
            model_src_dataset_name,
            target_transform_keys,
            build_target_transform,
        )
    else:

        dataset_transform_dir = os.path.join(
            transform_dir, active_dataset_name, input_transform_key
        )

        print(" >> >> INSIDE _find_or_create_transforms dataset_trasnform_dir", dataset_transform_dir)

        os.makedirs(dataset_transform_dir, exist_ok=True)
        input_transform_path = os.path.join(dataset_transform_dir, "input.pickle")
        target_transform_path = os.path.join(dataset_transform_dir, "target.pickle")

        lock_path = os.path.join(transform_dir, ".lock")
        lock = Lock(lock_path, lifetime=timedelta(hours=1))
        print(" >> >> INSIDE _find_or_create_transforms made_lock", lock_path)
        with lock:
            print(" <> >< <> >< <> >< <> >< <> >< <> >< <> >< <> >< <> >< <>")
            if os.path.exists(input_transform_path):
                start_time = time.time()
                print(" >> >> INSIDE data_utils._find_or_create_transforms: Loading input_transform")
                input_transform = load_transform(input_transform_path)
                end_time = time.time()
                print(f" >> >> INSIDE data_utils._find_or_create_transforms: Loaded input_transform, {end_time-start_time:.4f} seconds")
            else:
                start_time = time.time()
                print(" >> >> INSIDE data_utils._find_or_create_transforms: building input_transform")
                input_transform = _build_transform(
                    filename,
                    variables,
                    active_dataset_name,
                    model_src_dataset_name,
                    input_transform_key,
                    build_input_transform,
                )
                end_time = time.time()
                print(f" >> >> INSIDE data_utils._find_or_create_transforms: built input_transform, {end_time-start_time:.4f} seconds")
                save_transform(input_transform, input_transform_path)

            if os.path.exists(target_transform_path):
                start_time = time.time()
                print(" >> >> INSIDE data_utils._find_or_create_transforms: Loading target_transform")
                target_transform = load_transform(target_transform_path)
                end_time = time.time()
                print(f" >> >> INSIDE data_utils._find_or_create_transforms: Loaded target_transform, {end_time-start_time:.4f} seconds")
            else:
                if evaluation:
                    raise RuntimeError(
                        "Target transform should only be fitted during training"
                    )
                start_time = time.time()
                print(" >> >> INSIDE data_utils._find_or_create_transforms: building target_transform")
                target_transform = _build_transform(
                    filename,
                    target_variables,
                    active_dataset_name,
                    model_src_dataset_name,
                    target_transform_keys,
                    build_target_transform,
                )
                end_time = time.time()
                print(f" >> >> INSIDE data_utils._find_or_create_transforms: built target_transform, {end_time-start_time:.4f} seconds")
                save_transform(target_transform, target_transform_path)

    gc.collect
    return input_transform, target_transform
#====================================================================
def generate_output_filepath(output_dirpath):
    output_dir = Path(output_dirpath)

    # Check if the directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"The directory {output_dirpath} does not exist.")

    # Count the number of .nc files in the directory
    nc_files = list(output_dir.glob("*.nc"))
    count = len(nc_files)

    # Generate the output filepath with an incremented integer
    output_filepath = os.path.join(output_dir, "predictions-"+str(count)+".nc")

    return output_filepath
#====================================================================
TIME_RANGE = (
    datetime(2000, 6, 1),
    datetime(2024, 11, 30),
)
#====================================================================
def custom_collate(batch):
        return *default_collate([(e[0], e[1]) for e in batch]), np.concatenate(
            [e[2] for e in batch]
        )
#====================================================================
def _get_zarr_length(zarr_path):
    ds = xr.open_zarr(zarr_path, consolidated=True)
    n = len(ds.time)
    try:
        ds.close()
    except Exception:
        # ds.close() exists and will release any file handles. (docs). 
        pass
    return n
