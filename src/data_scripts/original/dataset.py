import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
#====================================================================
class DownscalingDataset(Dataset):
    def __init__(self, ds, variables, target_variables, time_range):
        self.ds = ds
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range

    @classmethod
    def variables_to_tensor(cls, ds, variables):
        return torch.tensor(
            # stack features before lat-lon (HW)
            np.stack([ds[var].astype("float32").values for var in variables], axis=-3)
        ).float()

    @classmethod
    def time_to_tensor(cls, ds, shape, time_range):
        #print(" >> >> INSIDE dataset ", type(ds["time"]), type(ds["time"].values), type(time_range), type(time_range[0]))

        start = np.datetime64(time_range[0])
        end = np.datetime64(time_range[1])
        
        # Compute total range in days
        delta_days = (end - start) / np.timedelta64(1, 'D')
        
        climate_time = np.array([(ds["time"].values - start) / np.timedelta64(1, 'D') / delta_days])
        #print(" >> >> INSIDE dataset climate_time", climate_time)

        #climate_time = np.array(ds["time"].values - time_range[0]) / np.array([time_range[1] - time_range[0]], dtype=np.dtype("timedelta64[ns]"))

        season_time = ds["time.dayofyear"].values / 360

        return (
            torch.stack(
                [
                    torch.tensor(climate_time).broadcast_to(
                        (climate_time.shape[0], *shape[-2:])
                    ),
                    torch.sin(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                    torch.cos(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                ],
                dim=-3,
            )
            .squeeze()
            .float()
        )

    def __len__(self):
        return len(self.ds.time) # * len(self.ds.ensemble_member)

    def __getitem__(self, idx):
        subds = self.sel(idx)

        cond = self.variables_to_tensor(subds, self.variables)
        if self.time_range is not None:
            cond_time = self.time_to_tensor(subds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        x = self.variables_to_tensor(subds, self.target_variables)

        time = subds["time"].values.reshape(-1)

        return cond, x, time

    def sel(self, idx):
        #em_idx, time_idx = divmod(idx, len(self.ds.time))
        #return self.ds.isel(time=time_idx, ensemble_member=em_idx)
        time_idx = idx
        return self.ds.isel(time=time_idx)