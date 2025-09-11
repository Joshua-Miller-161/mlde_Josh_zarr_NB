import sys
sys.dont_write_bytecode = True
import pickle
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path, PosixPath

from ..data_utils import is_main_process
#====================================================================
class DownscalingDataset(Dataset):
    def __init__(
        self,
        file_path,            # either xarray.Dataset (for val/test) OR path to .zarr (for training)
        variables,
        target_variables,
        time_range,
        _len
    ):
        self.file_path = file_path
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range
        self._len = _len

        # worker-local objects (will be created in worker process)
        self.ds = None

    def _ensure_open(self):
        if self.ds is not None:
            #if is_main_process():
            #    print(f" >> >> INSIDE file_path_only.dataset self.ds is not None: self._file_path {self._file_path}, ds {type(self.ds)}")
            return

        if isinstance(self.file_path, str) or isinstance(self.file_path, PosixPath):
            # open zarr inside worker process
            self.ds = xr.open_zarr(self.file_path, consolidated=True)
            #if is_main_process():
            #    print(f" >> >> INSIDE path_only.dataset isinstance str/Posixfile_path: self._file_path {self._file_path}, ds {type(self.ds)}")
        else:
            # parent supplied a materialized xarray dataset (val/test)
            print(f" >> >> ERROR INSIDE file_path_only.dataset else: ERROR self.file_path {self.file_path}, ds {type(self.ds)}")
            #

        # debug print to verify worker loaded resources
        import os as _os
        pid = _os.getpid()
        try:
            worker_info = None
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            wid = worker_info.id if worker_info is not None else "main"
        except Exception:
            wid = "unknown"
        #print(f"[DownscalingDataset] worker {wid} pid={pid} opened ds")

    @staticmethod
    def variables_to_tensor(ds, variables):
        return torch.tensor(
            np.stack([ds[var].astype("float32").values for var in variables], axis=-3)
        ).float()

    @staticmethod
    def time_to_tensor(ds, shape, time_range):
        start = np.datetime64(time_range[0])
        end = np.datetime64(time_range[1])
        delta_days = (end - start) / np.timedelta64(1, "D")
        climate_time = np.array([(ds["time"].values - start) / np.timedelta64(1, "D") / delta_days])
        season_time = ds["time.dayofyear"].values / 360.0
        return (
            torch.stack(
                [
                    torch.tensor(climate_time).broadcast_to((climate_time.shape[0], *shape[-2:])),
                    torch.sin(2 * np.pi * torch.tensor(season_time).broadcast_to((climate_time.shape[0], *shape[-2:]))),
                    torch.cos(2 * np.pi * torch.tensor(season_time).broadcast_to((climate_time.shape[0], *shape[-2:]))),
                ],
                dim=-3,
            )
            .squeeze()
            .float()
        )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._ensure_open()
        return self.ds.isel(time=idx)