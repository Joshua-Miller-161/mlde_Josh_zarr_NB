import sys
sys.dont_write_bytecode = True
import pickle
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path, PosixPath
import zarr

from ..data_utils import is_main_process, decode_zarr_time_array
#====================================================================
class DownscalingDataset(Dataset):
    def __init__(self, file_path, variables, target_variables, time_range, _len):
        self.file_path = str(file_path)  # ensure string
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range
        self._len = _len

        # worker-local objects (created lazily)
        self.opened = False

    def _ensure_open(self):
        if self.opened:
            return

        # open zarr once per worker
        self.z = zarr.open_consolidated(self.file_path)

        # keep references to arrays
        self.var_arrays = {v: self.z[v] for v in self.variables}
        self.target_arrays = {v: self.z[v] for v in self.target_variables}

        # pre-read time array
        if "time" in self.z.array_keys():
            self.time_values = decode_zarr_time_array(self.z, time_key="time")
        else:
            self.time_values = None

        self.opened = True

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._ensure_open()
        # read all input variables
        cond_list = [self.var_arrays[v][idx] for v in self.variables]
        target_list = [self.target_arrays[v][idx] for v in self.target_variables]

        # stack into arrays
        cond = np.stack(cond_list, axis=0).astype("float32")
        target = np.stack(target_list, axis=0).astype("float32")

        # time
        time_value = self.time_values[idx] if self.time_values is not None else None

        return cond, target, time_value

    @staticmethod
    def np_to_tensor(arr):
        return torch.tensor(arr).float()

    @staticmethod
    def time_to_tensor(time_values, batch_shape, time_range):
        if time_values is None:
            return None
        B = batch_shape[0]
        H = batch_shape[-2]
        W = batch_shape[-1]
        start = np.datetime64(time_range[0])
        end = np.datetime64(time_range[1])
        delta_days = (end - start) / np.timedelta64(1, "D")
        climate_time = ((time_values - start) / np.timedelta64(1, "D")) / delta_days  # (B,)
        climate_ch = np.broadcast_to(climate_time.reshape(B,1,1), (B,1,H,W))
        doy = (time_values.astype('datetime64[D]').view('int64') % 365) / 360.0  # (B,)
        sin_ch = np.broadcast_to(np.sin(2*np.pi*doy).reshape(B,1,1), (B,1,H,W))
        cos_ch = np.broadcast_to(np.cos(2*np.pi*doy).reshape(B,1,1), (B,1,H,W))
        out_np = np.concatenate([climate_ch, sin_ch, cos_ch], axis=1)  # (B,3,H,W)
        return torch.tensor(out_np, dtype=torch.float32)

    # def time_to_tensor(time_values, shape, time_range):
    #     if time_values is None:
    #         return None
    #     start = np.datetime64(time_range[0])
    #     end = np.datetime64(time_range[1])
    #     delta_days = (end - start) / np.timedelta64(1, "D")
    #     climate_time = ((time_values - start) / np.timedelta64(1, "D") / delta_days).reshape(-1, 1, 1)
    #     season_time = (time_values.astype('datetime64[D]').view('int64') % 365) / 360.0
    #     season_time = season_time.reshape(-1, 1, 1)
    #     # broadcast to spatial dimensions
    #     return torch.tensor(np.concatenate([
    #         np.broadcast_to(climate_time, shape),
    #         np.sin(2*np.pi*season_time).repeat(shape[-2], axis=1).repeat(shape[-1], axis=2),
    #         np.cos(2*np.pi*season_time).repeat(shape[-2], axis=1).repeat(shape[-1], axis=2),
    #     ], axis=0), dtype=torch.float32)


# class DownscalingDataset(Dataset):
#     def __init__(
#         self,
#         file_path,            # either xarray.Dataset (for val/test) OR path to .zarr (for training)
#         variables,
#         target_variables,
#         time_range,
#         _len
#     ):
#         self.file_path = file_path
#         self.variables = variables
#         self.target_variables = target_variables
#         self.time_range = time_range
#         self._len = _len

#         # worker-local objects (will be created in worker process)
#         self.ds = None

#     def _ensure_open(self):
#         if self.ds is not None:
#             #if is_main_process():
#             #    print(f" >> >> INSIDE file_path_only.dataset self.ds is not None: self._file_path {self._file_path}, ds {type(self.ds)}")
#             return

#         if isinstance(self.file_path, str) or isinstance(self.file_path, PosixPath):
#             # open zarr inside worker process
#             self.ds = xr.open_zarr(self.file_path, consolidated=True)
#             #if is_main_process():
#             #    print(f" >> >> INSIDE path_only.dataset isinstance str/Posixfile_path: self._file_path {self._file_path}, ds {type(self.ds)}")
#         else:
#             # parent supplied a materialized xarray dataset (val/test)
#             print(f" >> >> ERROR INSIDE file_path_only.dataset else: ERROR self.file_path {self.file_path}, ds {type(self.ds)}")
#             #

#         # debug print to verify worker loaded resources
#         import os as _os
#         pid = _os.getpid()
#         try:
#             worker_info = None
#             from torch.utils.data import get_worker_info
#             worker_info = get_worker_info()
#             wid = worker_info.id if worker_info is not None else "main"
#         except Exception:
#             wid = "unknown"
#         #print(f"[DownscalingDataset] worker {wid} pid={pid} opened ds")

#     @staticmethod
#     def variables_to_tensor(ds, variables):
#         return torch.tensor(
#             np.stack([ds[var].astype("float32").values for var in variables], axis=-3)
#         ).float()

#     @staticmethod
#     def time_to_tensor(ds, shape, time_range):
#         start = np.datetime64(time_range[0])
#         end = np.datetime64(time_range[1])
#         delta_days = (end - start) / np.timedelta64(1, "D")
#         climate_time = np.array([(ds["time"].values - start) / np.timedelta64(1, "D") / delta_days])
#         season_time = ds["time.dayofyear"].values / 360.0
#         return (
#             torch.stack(
#                 [
#                     torch.tensor(climate_time).broadcast_to((climate_time.shape[0], *shape[-2:])),
#                     torch.sin(2 * np.pi * torch.tensor(season_time).broadcast_to((climate_time.shape[0], *shape[-2:]))),
#                     torch.cos(2 * np.pi * torch.tensor(season_time).broadcast_to((climate_time.shape[0], *shape[-2:]))),
#                 ],
#                 dim=-3,
#             )
#             .squeeze()
#             .float()
#         )

#     def __len__(self):
#         return self._len

#     def __getitem__(self, idx):
#         self._ensure_open()
#         return self.ds.isel(time=idx)