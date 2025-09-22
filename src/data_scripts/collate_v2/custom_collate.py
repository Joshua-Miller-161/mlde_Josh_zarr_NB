import sys
sys.dont_write_bytecode = True
import os
import xarray as xr
import torch
import logging
import torch.distributed as dist
import time as clock

logger = logging.getLogger()

from .dataset import DownscalingDataset
from ..data_utils import is_main_process
#====================================================================
class TransformCollateFn:
    def __init__(self, input_transform, target_transform, variables, target_variables, time_range):
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range

    def __call__(self, batch):
        # batch is a list of xarray.Dataset objects

        start_time = clock.time()
        batch_ds = xr.concat(batch, dim="time")
        end_time = clock.time()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        logger.info(" >> >> INSIDE collate_v2.custom_collate [rank %d pid %d] concat data: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))


        start_time = clock.time()
        if self.input_transform is not None:
            batch_ds = self.input_transform.transform(batch_ds)

        if self.target_transform is not None:
            batch_ds = self.target_transform.transform(batch_ds)
        end_time = clock.time()

        # for var_name in batch_ds.data_vars:
        #     mean_ = batch_ds[var_name].mean().compute().values
        #     min_  = batch_ds[var_name].min().compute().values
        #     max_  = batch_ds[var_name].max().compute().values
        #     print(" >> >> INSIDE collate_v2.custom_collate -", var_name, ", mean =", mean_, ", min =", min_, ", max =", max_)


        if is_main_process():
            print(" >> >> INSIDE collate_v2.custom_collate ")
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        logger.info(" >> >> INSIDE collate_v2.custom_collate [rank %d pid %d] transform data: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))

        cond = DownscalingDataset.variables_to_tensor(batch_ds, self.variables)
        x = DownscalingDataset.variables_to_tensor(batch_ds, self.target_variables)

        ###### PROBABLY WRONG #####
        if self.time_range is not None:
            cond_time = DownscalingDataset.time_to_tensor(batch_ds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        time = batch_ds["time"].values.reshape(-1)
        return cond, x, time