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
        data_or_path,            # either xarray.Dataset (for val/test) OR path to .zarr (for training)
        variables,
        target_variables,
        time_range,
        input_transform_path,
        target_transform_path,
        _len
    ):
        self._data_or_path = data_or_path
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range
        self.input_transform_path = input_transform_path
        self.target_transform_path = target_transform_path
        self._len = _len

        # worker-local objects (will be created in worker process)
        self.ds = None
        self.input_transform = None
        self.target_transform = None

    def _load_transforms_from_disk(self):
        if self.input_transform_path and self.input_transform is None:
            with open(self.input_transform_path, "rb") as f:
                self.input_transform = pickle.load(f)

        if self.target_transform_path and self.target_transform is None:
            with open(self.target_transform_path, "rb") as f:
                self.target_transform = pickle.load(f)

    def _ensure_open(self):
        if self.ds is not None:
            #if is_main_process():
            #    print(f" >> >> INSIDE path_only.dataset self.ds is not None: self._data_or_path {self._data_or_path}, ds {type(self.ds)}")
            return

        if isinstance(self._data_or_path, str) or isinstance(self._data_or_path, PosixPath):
            # open zarr inside worker process
            self.ds = xr.open_zarr(self._data_or_path, consolidated=True)
            #if is_main_process():
            #    print(f" >> >> INSIDE path_only.dataset isinstance str/PosixPath: self._data_or_path {self._data_or_path}, ds {type(self.ds)}")
        else:
            # parent supplied a materialized xarray dataset (val/test)
            #print(f" >> >> INSIDE path_only.dataset else: self._data_or_path {self._data_or_path}, ds {type(self.ds)}")
            self.ds = self._data_or_path

        # load transforms in this worker after DS opened
        self._load_transforms_from_disk()

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
        #print(f"[DownscalingDataset] worker {wid} pid={pid} opened ds and loaded transforms")

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
        subds = self.ds.isel(time=idx)

        # apply fitted transforms (already saved in pickles)
        if self.input_transform is not None:
            # your transform API used transform.transform(xr_dataset)
            subds = self.input_transform.transform(subds)

        if self.target_transform is not None:
            subds = self.target_transform.transform(subds)

        cond = self.variables_to_tensor(subds, self.variables)
        if self.time_range is not None:
            cond_time = self.time_to_tensor(subds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        x = self.variables_to_tensor(subds, self.target_variables)
        time = subds["time"].values.reshape(-1)
        return cond, x, time