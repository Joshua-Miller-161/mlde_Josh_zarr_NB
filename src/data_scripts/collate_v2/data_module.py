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
from ..data_utils import custom_collate, TIME_RANGE, get_variables, is_main_process, _get_zarr_length
from .custom_collate import TransformCollateFn
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
        active_dataset_name,
        model_src_dataset_name,
        input_transform_dataset_name,
        input_transform_key,
        target_transform_keys,
        transform_dir,
        batch_size,
        filename,
        include_time_inputs=True,
        evaluation=False,
        shuffle=True,
        num_workers=0,
        prefetch_factor=0
    ):
        super().__init__()
        self.active_dataset_name = active_dataset_name
        self.model_src_dataset_name = model_src_dataset_name
        self.input_transform_dataset_name = input_transform_dataset_name
        self.input_transform_key = input_transform_key
        self.target_transform_keys = target_transform_keys
        self.transform_dir = transform_dir
        self.filename = filename
        self.batch_size = batch_size
        self.include_time_inputs = include_time_inputs
        self.evaluation = evaluation
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor 

        self.time_range = TIME_RANGE if self.include_time_inputs else None

        self.variables, self.target_variables = get_variables(model_src_dataset_name)

        self.train_data = 69
        self.val_data = 69
        self.test_data = 69
        self.train_transform = 69
        self.train_target_transform = 69
        self.test_transform = 69
        self.test_target_transform = 69

        self.dataset_transform_dir = os.path.join(
            self.transform_dir, self.active_dataset_name, self.input_transform_key
        )

        self.train_len = 69
        self.val_len = 69
        self.collate_class = 69
        self.test_collate_class = 69

    def setup(self, stage=None):
        if is_main_process():
            print(" >> >> inside lightningDataModule.setup")
        logger.info(" >> >> inside lightningDataModule.setup")
        if stage == "fit" or stage is None:
            # For TRAIN: do NOT materialize. Keep transforms.
            self.train_zarr_path, self.train_transform, self.train_target_transform = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.input_transform_key,
                self.target_transform_keys,
                self.transform_dir,
                self.filename
            )
            self.train_len = _get_zarr_length(self.train_zarr_path)

            # For VAL: fine to materialize because you use num_workers=0
            self.val_zarr_path, _, _ = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.input_transform_key,
                self.target_transform_keys,
                self.transform_dir,
                "val_consolodated.zarr",
            )
            self.val_len = _get_zarr_length(self.val_zarr_path)

            self.collate_class = TransformCollateFn(
                input_transform=self.train_transform,
                target_transform=self.train_target_transform,
                variables=self.variables,
                target_variables=self.target_variables,
                time_range=self.time_range
            )

        if stage == "test" or stage is None:
            self.test_zarr_path, self.test_transform, self.test_target_transform = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.input_transform_key,
                self.target_transform_keys,
                self.transform_dir,
                self.filename,
                evaluation=self.evaluation,    # keep 0 workers here too
            )
            self.test_len = _get_zarr_length(self.test_zarr_path)

            self.test_collate_class = TransformCollateFn(
                input_transform=self.train_transform,
                target_transform=self.train_target_transform,
                variables=self.variables,
                target_variables=self.target_variables,
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

        data_loader = DataLoader(
            xr_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_class,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            **({"prefetch_factor": self.prefetch_factor} if self.num_workers > 0 else {}),
            **({"multiprocessing_context":ctx} if self.num_workers > 0 else {})
        )
        return data_loader

    def val_dataloader(self):
        xr_dataset = DownscalingDataset(
            self.val_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.val_len
        )

        data_loader = DataLoader(
            xr_dataset,
            batch_size=self.batch_size,
            shuffle=False, #(self.shuffle and sampler is None),
            collate_fn=self.collate_class,
            num_workers=0
        )
        return data_loader

    def test_dataloader(self):
        xr_dataset = DownscalingDataset(
            self.test_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.test_len
        )

        data_loader = DataLoader(
            xr_dataset,
            batch_size=self.batch_size,
            shuffle=False, #(self.shuffle and sampler is None),
            collate_fn=self.test_collate_class,
            num_workers=0
        )
        return data_loader