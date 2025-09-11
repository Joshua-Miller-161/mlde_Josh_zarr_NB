import sys
sys.dont_write_bytecode = True
import xarray as xr
import logging
import torch.distributed as dist
import time

logger = logging.getLogger()

from ..data_utils import is_main_process, _find_or_create_transforms, datafile_path, open_zarr
#====================================================================
def get_xr_dataset(
    active_dataset_name,
    model_src_dataset_name,
    input_transform_dataset_name,
    input_transform_key,
    target_transform_keys,
    transform_dir,
    filename,
    evaluation=False,
    materialize: bool = True,   # <— NEW: True = old behavior
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

    transform, target_transform = _find_or_create_transforms(
        filename,
        input_transform_dataset_name,
        model_src_dataset_name,
        transform_dir,
        input_transform_key,
        target_transform_keys,
        evaluation
    )

    if materialize:
        start_time = time.time()
        xr_data = open_zarr(active_dataset_name, filename)
        end_time = time.time()
        if is_main_process():
            print(f" <> <> mlde_josh_utils.data.data_utils.get_xr_dataset reading xr_data: {end_time - start_time:.4f} seconds")
            print(f" >> >> mlde_josh_utils.data.data_utils.get_xr_dataset loaded_dataset {filename}")
        logger.info(" <> <> mlde_josh_utils.data.data_utils.get_xr_dataset reading xr_data: %.4f seconds", end_time - start_time)
        logger.info(" >> >> mlde_josh_utils.data.data_utils.get_xr_dataset loaded_dataset %s", str(filename))

        start_time = time.time()
        xr_data = transform.transform(xr_data)
        xr_data = target_transform.transform(xr_data)
        end_time = time.time()
        if is_main_process():
            print(f" <> <> mlde_josh_utils.data.data_utils.get_xr_dataset transform xr_data: {end_time - start_time:.4f} seconds")
            print(" >> >> mlde_josh_utils.data.data_utils.get_xr_dataset fitted transform")
        logger.info(" <> <> mlde_josh_utils.data.data_utils.get_xr_dataset transform xr_data: %.4f seconds", end_time - start_time)
        logger.info(" >> >> mlde_josh_utils.data.data_utils.get_xr_dataset fitted transform")

        return xr_data, transform, target_transform

    # materialize == False: return only a path for worker-side opening
    zarr_path = datafile_path(active_dataset_name, filename)
    if is_main_process():
        print(f" >> >> mlde_josh_utils.data.data_utils.get_xr_dataset returning path only: {zarr_path}")
    return zarr_path, transform, target_transform