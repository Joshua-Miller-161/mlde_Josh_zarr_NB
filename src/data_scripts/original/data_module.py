import sys
sys.dont_write_bytecode = True
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.distributed as dist
import logging

logger = logging.getLogger()

from .dataset import DownscalingDataset
from ..data_utils import get_xr_dataset, custom_collate, TIME_RANGE, get_variables, is_main_process
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
        filename='train.nc',
        include_time_inputs=True,
        evaluation=False,
        shuffle=True,
        num_workers=0
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

        self.time_range = TIME_RANGE if self.include_time_inputs else None

        self.variables, self.target_variables = get_variables(model_src_dataset_name)

        self.train_data = 69
        self.val_data = 69
        self.test_data = 69
        self.train_transform = 69
        self.train_target_transform = 69
        self.test_transform = 69
        self.test_target_transform = 69

    def setup(self, stage=None):
        if is_main_process():
            print(" >> >> inside lightningDataModule.setup")
        logger.info(" >> >> inside lightningDataModule.setup")
        if stage == "fit" or stage is None:
            self.train_data, self.train_transform, self.train_target_transform = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.input_transform_key,
                self.target_transform_keys,
                self.transform_dir,
                self.filename,
            )
            if is_main_process():
                print(" >> >> finished lightningDataModule.setup fit", type(self.train_data), type(self.train_transform), type(self.train_target_transform))
            logger.info(" >> >> finished lightningDataModule.setup fit %s %s %s", type(self.train_data), type(self.train_transform), type(self.train_target_transform))
            self.val_data, _, _ = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.input_transform_key,
                self.target_transform_keys,
                self.transform_dir,
                "val_consolodated.zarr" #val.nc
            )

        if stage == "test" or stage is None:
            self.test_data, self.test_transform, self.test_target_transform = get_xr_dataset(
                self.active_dataset_name,
                self.model_src_dataset_name,
                self.input_transform_dataset_name,
                self.input_transform_key,
                self.target_transform_keys,
                self.transform_dir,
                self.filename,
                self.evaluation
            )

    def train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        
        if is_main_process():
            print(" >> >> inside lightningDataModule.train_dataloader", type(self.train_data))
        logger.info(" >> >> inside lightningDataModule.train_dataloader [Rank %d]: %s", rank, type(self.train_data))
        
        # keep workers modest; oversubscription hurts I/O
        # num_workers = 0 #min(4, max(2, (os.cpu_count() or 8) // max(1, world_size)))

        xr_dataset = DownscalingDataset(
            self.train_data,
            self.variables,
            self.target_variables,
            self.time_range
        )

        data_loader = DataLoader(
            xr_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle, #(self.shuffle and sampler is None),
            #sampler=sampler,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,  # keeps shards even across ranks
            **({"prefetch_factor": self.num_workers} if self.num_workers > 0 else {})
        )
        return data_loader

    def val_dataloader(self):
        xr_dataset = DownscalingDataset(
            self.val_data,
            self.variables,
            self.target_variables,
            self.time_range
        )

        data_loader = DataLoader(
            xr_dataset,
            batch_size=self.batch_size,
            shuffle=False, #(self.shuffle and sampler is None),
            collate_fn=custom_collate,
            num_workers=0
        )
        return data_loader

    def test_dataloader(self):
        xr_dataset = DownscalingDataset(
            self.test_data,
            self.variables,
            self.target_variables,
            self.time_range
        )

        data_loader = DataLoader(
            xr_dataset,
            batch_size=self.batch_size,
            shuffle=False, #(self.shuffle and sampler is None),
            collate_fn=custom_collate,
            num_workers=0
        )
        return data_loader




# @staticmethod
# def _wrap_loader(xr_data, model_src_dataset_name, batch_size, shuffle, include_time_inputs):
#     # sampler = DistributedSampler(dataset) if self.trainer and self.trainer.world_size > 1 else None
#     if is_main_process():
#         print(" >> >> inside lightningDataModule._wrap_loader", type(xr_data))
#     logger.info(" >> >> inside lightningDataModule._wrap_loader %s", type(xr_data))

#     time_range = None
#     if include_time_inputs:
#         time_range = TIME_RANGE

#     variables, target_variables = get_variables(model_src_dataset_name)

#     xr_dataset = DownscalingDataset(xr_data,
#                                     variables,
#                                     target_variables,
#                                     time_range)

#     data_loader = DataLoader(xr_dataset,
#                                 batch_size=batch_size,
#                                 shuffle=shuffle,
#                                 collate_fn=custom_collate)
#     return data_loader


# def build_dataloader(
#     xr_data, variables, target_variables, batch_size, shuffle, include_time_inputs
# ):
#     def custom_collate(batch):
#         from torch.utils.data import default_collate

#         return *default_collate([(e[0], e[1]) for e in batch]), np.concatenate(
#             [e[2] for e in batch]
#         )

#     print(" >> INSIDE data.build_dataloader")
#     time_range = None
#     if include_time_inputs:
#         time_range = TIME_RANGE
#     xr_dataset = UKCPLocalDataset(xr_data, variables, target_variables, time_range)
#     data_loader = DataLoader(
#         xr_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate
#     )
#     print(" >> >> DONE build_dataloader")
#     return data_loader