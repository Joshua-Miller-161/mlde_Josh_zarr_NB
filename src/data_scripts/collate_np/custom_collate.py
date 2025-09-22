import sys
sys.dont_write_bytecode = True
import os
import xarray as xr
import torch
import logging
import torch.distributed as dist
import time as clock
import numpy as np

logger = logging.getLogger()

from .dataset import DownscalingDataset
from ..data_utils import is_main_process
#====================================================================
class FastCollate:
    def __init__(self, input_transform=None, target_transform=None, time_range=None):
        self.input_transform = input_transform  # should operate on numpy arrays or be None
        self.target_transform = target_transform
        self.time_range = time_range

    def __call__(self, batch):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        start_time = clock.time()
        # batch is list of (cond_np, targ_np, time)
        conds = np.stack([b[0] for b in batch], axis=0)  # (B, C, H, W)
        targs = np.stack([b[1] for b in batch], axis=0)
        times = np.array([b[2] for b in batch])
        end_time = clock.time()
        logger.info(" >> >> INSIDE collate_np.FastCollate [rank %d pid %d] concat data: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))


        start_time = clock.time()
        if self.input_transform is not None:
            # ensure transform works on numpy arrays (vectorized)
            conds = self.input_transform.transform(conds)
        end_time = clock.time()
        logger.info(" >> >> INSIDE collate_np.FastCollate [rank %d pid %d] input transform: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))

        start_time = clock.time()
        if self.target_transform is not None:
            targs = self.target_transform.transform(targs)
        end_time = clock.time()
        logger.info(" >> >> INSIDE collate_np.FastCollate [rank %d pid %d] target transform: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))

        # convert to torch
        conds = torch.from_numpy(conds)
        targs = torch.from_numpy(targs)

        # for i in range(conds.shape[1]):
        #    logger.info(" > - > - INSIDE FastCollate conds mean = %.6f, std = %.6f, max = %.6f, min = %.6f", torch.mean(conds[0][i]).cpu().detach().numpy(), torch.std(conds[0][i]).cpu().detach().numpy(), torch.max(conds[0][i]).cpu().detach().numpy(), torch.min(conds[0][i]).cpu().detach().numpy())
        # for j in range(targs.shape[1]):
        #     logger.info(" > - > - INSIDE FastCollate targs mean = %.6f, std = %.6f, max = %.6f, min = %.6f", torch.mean(targs[0][j]).cpu().detach().numpy(), torch.std(targs[0][j]).cpu().detach().numpy(), torch.max(targs[0][j]).cpu().detach().numpy(), torch.min(targs[0][j]).cpu().detach().numpy())
        
        #if is_main_process():
        #    print(" > - > - INSIDE FastColllate conds.shape", conds.shape)
        #    print(" > - > - INSIDE FastColllate targs.shape", targs.shape)
        #logger.info(" >> >> INSIDE collate_np.FastCollate [rank %d pid %d] conds.shape: %s", rank, os.getpid(), str(conds.shape))
        #logger.info(" >> >> INSIDE collate_np.FastCollate [rank %d pid %d] conds.shape: %s", rank, os.getpid(), str(targs.shape))

        ###### PROBABLY WRONG #####
        if self.time_range is not None:
            cond_time = DownscalingDataset.time_to_tensor(times, cond.shape, self.time_range)

            #if is_main_process():
            #    print(" > - > - INSIDE FastColllate cond_time.shape", cond_time.shape)
            logger.info(" >> >> INSIDE collate_np.FastCollate [rank %d pid %d] conds.shape: %s", rank, os.getpid(), str(cond_time.shape))
            
            cond = torch.cat([cond, cond_time])
        
        return conds, targs, times

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