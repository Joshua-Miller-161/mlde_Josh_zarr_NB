import sys
sys.dont_write_bytecode = True
import xarray as xr
import torch
import logging

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
        batch_ds = xr.concat(batch, dim="time")

        if self.input_transform is not None:
            batch_ds = self.input_transform.transform(batch_ds)

        if self.target_transform is not None:
            batch_ds = self.target_transform.transform(batch_ds)

        if is_main_process():
            print(" >> >> INSIDE collate_v2.custom_collate ")
        cond = DownscalingDataset.variables_to_tensor(batch_ds, self.variables)
        x = DownscalingDataset.variables_to_tensor(batch_ds, self.target_variables)

        ###### PROBABLY WRONG #####
        if self.time_range is not None:
            cond_time = DownscalingDataset.time_to_tensor(batch_ds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        time = batch_ds["time"].values.reshape(-1)
        return cond, x, time