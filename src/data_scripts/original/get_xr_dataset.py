import sys
sys.dont_write_bytecode = True
import xarray as xr
import logging
import torch.distributed as dist
import time

logger = logging.getLogger()

from ..data_utils import is_main_process, _find_or_create_transforms, datafile_path
#====================================================================
def get_xr_dataset(active_dataset_name,
                   model_src_dataset_name,
                   input_transform_dataset_name,
                   input_transform_key,
                   target_transform_keys,
                   transform_dir,
                   filename,
                   evaluation=False):
    """Get xarray.Dataset and transforms for a named dataset split.

    Args:
      active_dataset_name: Name of dataset from which to load data splits
      model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
      input_transform_dataset_name: Name of dataset to use for fitting input transform (may be the same as active_dataset_name or model_src_dataset_name)
      input_transform_key: Name of input transform pipeline to use
      target_transform_keys: Mapping from name of target variable to name of target transform pipeline to use
      transform_dir: Path to where transforms should be stored
      filename: Split of the active dataset to load
      ensemble_members: Ensemble members to load
      evaluation: If `True`, don't allow fitting of target transform

    Returns:
      dataset, transform, target_transform
    """

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    if is_main_process():
        print(f" >> >> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset [Rank {rank}]")
    logger.info(" >> >> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset [Rank %d]", rank)

    transform, target_transform = _find_or_create_transforms(
        filename,
        input_transform_dataset_name,
        model_src_dataset_name,
        transform_dir,
        input_transform_key,
        target_transform_keys,
        evaluation
    )
    
    start_time = time.time()
    xr_data = open_zarr(active_dataset_name, filename)
    end_time = time.time()
    
    if is_main_process():
        print(f" <> <> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset [Rank {rank}] reading xr_data: {end_time - start_time:.4f} seconds")
        print(f" >> >> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset [Rank {rank}] loaded_dataset", str(filename))
    logger.info(" <> <> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset [Rank %d] reading xr_data: %.4f seconds", rank, end_time - start_time)
    logger.info(" >> >> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset [Rank %d] loaded_dataset %s", rank, str(filename))

    start_time = time.time()
    xr_data = transform.transform(xr_data)
    xr_data = target_transform.transform(xr_data)
    end_time = time.time()
    
    if is_main_process():
        print(f" <> <> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset transform xr_data: {end_time - start_time:.4f} seconds")
        print(" >> >> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset fitted transform")

    logger.info(" <> <> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset [Rank %d] transform xr_data: %.4f seconds", rank, end_time - start_time)
    logger.info(" >> >> INSIDE mlde_josh_utils.data.data_utils.get_xr_dataset fitted transform")

    return xr_data, transform, target_transform