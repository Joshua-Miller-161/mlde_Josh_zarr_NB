import sys
sys.dont_write_bytecode = True
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.distributed as dist
import logging
import multiprocessing as mp

logger = logging.getLogger()

from .dataset import DownscalingDataset
from .get_xr_dataset import get_xr_dataset
from .custom_collate import FastCollate
from ..data_utils import TIME_RANGE, get_variables_per_var, is_main_process, _get_zarr_length
#====================================================================
def _worker_init_fn(worker_id):
    # limit threads to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # set a small file cache (must be > 0) or skip entirely
    #xr.set_options(file_cache_maxsize=1, warn_on_unclosed_files=True)

ctx = mp.get_context("spawn")
#====================================================================
class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        active_dataset_name,
        model_src_dataset_name,
        input_transform_dataset_name,
        transform_dir,
        batch_size,
        filename,
        val_filename=None,
        include_time_inputs=True,
        evaluation=False,
        shuffle=True,
        num_workers=0,
        prefetch_factor=None
    ):
        super().__init__()
        self.config = config
        self.active_dataset_name = active_dataset_name
        self.model_src_dataset_name = model_src_dataset_name
        self.input_transform_dataset_name = input_transform_dataset_name
        self.transform_dir = transform_dir
        self.filename = filename
        self.val_filename = val_filename
        self.batch_size = batch_size
        self.include_time_inputs = include_time_inputs
        self.evaluation = evaluation
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor 

        self.time_range = TIME_RANGE if self.include_time_inputs else None

        self.variables, self.target_variables = get_variables_per_var(config)

        self.train_data = 69
        self.val_data = 69
        self.test_data = 69
        # self.train_transform = 69
        # self.train_target_transform = 69
        # self.test_transform = 69
        # self.test_target_transform = 69

        self.train_len = 69
        self.val_len = 69

        # just above DataLoader call: build kwargs robustly
        self.dl_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=None, # Will be set later
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
        )

        if self.num_workers > 0:
            # set multiprocessing context
            self.dl_kwargs["multiprocessing_context"] = ctx

            # decide prefetch factor:
            # - if user left prefetch_factor as None => set a sensible default (2)
            # - if user provided a positive integer => use it
            # - if user provided invalid value (<=0) => coerce to default
            default_pf = 2
            pf_user = self.prefetch_factor
            if pf_user is None:
                pf = default_pf
            else:
                try:
                    pf = int(pf_user)
                    if pf <= 0:
                        pf = default_pf
                except Exception:
                    pf = default_pf

            self.dl_kwargs["prefetch_factor"] = pf

    def setup(self, stage=None):
        if is_main_process():
            print(" >> >> inside lightningDataModule.setup")
        logger.info(" >> >> inside lightningDataModule.setup")
        if stage == "fit" or stage is None:
            # For TRAIN: do NOT materialize. Keep transforms.
            self.train_zarr_path, self.train_transforms, self.train_target_transforms = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.config,
                self.transform_dir,
                self.filename
            )
            self.train_len = _get_zarr_length(self.train_zarr_path)

            # For VAL: fine to materialize because you use num_workers=0
            self.val_zarr_path, _, _ = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.config,
                self.transform_dir,
                self.val_filename,
            )
            self.val_len = _get_zarr_length(self.val_zarr_path)

            self.dl_kwargs['collate_fn'] = FastCollate(
                input_transforms=self.train_transforms,
                target_transforms=self.train_target_transforms,
                time_range=self.time_range
            )


        if stage == "test" or stage is None:
            print(" >> >> INSIDE data_module setup <<TEST>>")
            self.test_zarr_path, self.test_transforms, self.test_target_transforms = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.config,
                self.transform_dir,
                self.filename,
                evaluation=self.evaluation,    # keep 0 workers here too
            )
            print(" >> >> INSIDE data_module setup <<TEST>> got_xr_dataset")

            self.test_len = _get_zarr_length(self.test_zarr_path)
            print(" >> >> INSIDE data_module setup <<TEST>> zarr_len =", self.test_len)

            self.dl_kwargs['collate_fn'] = FastCollate(
                input_transforms=self.test_transforms,
                target_transforms=self.test_target_transforms,
                time_range=self.time_range
            )

    def train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        
        if is_main_process():
            print(" >> >> inside lightningDataModule.train_dataloader", type(self.train_data))
        logger.info(" >> >> inside lightningDataModule.train_dataloader [Rank %d]: %s", rank, type(self.train_data))
        
        # keep workers modest; oversubscription hurts I/O
        # num_workers = 0 #min(4, max(2, (os.cpu_count() or 8) // max(1, world_size)))
        # spawn context recommended (shown previously)

        xr_dataset = DownscalingDataset(
            self.train_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.train_len
        )

        data_loader = DataLoader(xr_dataset, **self.dl_kwargs)
        
        return data_loader

    def val_dataloader(self):
        xr_dataset = DownscalingDataset(
            self.val_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.val_len
        )

        self.dl_kwargs['shuffle'] = False
        
        data_loader = DataLoader(xr_dataset, **self.dl_kwargs)
        
        return data_loader

    def test_dataloader(self):
        xr_dataset = DownscalingDataset(
            self.test_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.test_len
        )

        self.dl_kwargs['num_workers'] = 0
        self.dl_kwargs['shuffle'] = False

        data_loader = DataLoader(xr_dataset, **self.dl_kwargs)
        
        return data_loader