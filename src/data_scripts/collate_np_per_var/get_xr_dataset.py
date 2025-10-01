import sys
sys.dont_write_bytecode = True
import xarray as xr
import logging
import torch.distributed as dist
import time

logger = logging.getLogger()

from ..data_utils import is_main_process, datafile_path
from ...transforms_np import _find_or_create_transforms_per_variable_from_config
#====================================================================
def get_xr_dataset(
    active_dataset_name,
    model_src_dataset_name,
    input_transform_dataset_name,
    config,
    transform_dir,
    filename,
    evaluation=False,   # <— NEW: True = old behavior
):
    """
    Returns:
      if materialize:
         (xr_dataset, transform, target_transform)
      else:
         (zarr_path_str, transform, target_transform)
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if is_main_process():
        print(f" >> >> INSIDE ...get_xr_dataset [Rank {rank}]")
    logger.info(" >> >> INSIDE ...get_xr_dataset [Rank %d]", rank)

    input_transforms, target_transforms = _find_or_create_transforms_per_variable_from_config(
        filename,
        input_transform_dataset_name,
        model_src_dataset_name,
        transform_dir,
        config,
        evaluation
    )

    zarr_path = datafile_path(active_dataset_name, filename)
    if is_main_process():
        print(f" >> >> mlde_josh_utils.data.data_utils.get_xr_dataset returning path only: {zarr_path}")
    return zarr_path, input_transforms, target_transforms