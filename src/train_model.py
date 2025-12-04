"""Training for score-based generative models. """

import sys
sys.dont_write_bytecode = True
from collections import defaultdict
import os
from absl import flags
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from dotenv import load_dotenv
import logging
import torch
from torchinfo import summary
torch.set_float32_matmul_precision('medium')
import torchvision

logger = logging.getLogger(__name__)
#===================================================================
FLAGS = flags.FLAGS
#====================================================================
from src import cncsnpp
from src.deterministic_models import cncsnpp
from src.lightningModuleEMA import ScoreModelLightningModule
from src.utils import LossOnlyProgressBar, setup_checkpoint, check_saved_checkpoint, is_main_process, create_model, save_config
#from src.data_scripts.collate_np.data_module import LightningDataModule
from src.data_scripts.collate_np_per_var.data_module import LightningDataModule
#====================================================================
def train(config, workdir, filename, val_filename):
    
    load_dotenv()

    if ((config.deterministic == 'True') or (config.deterministic == 'true') or (config.deterministic == True) or (config.deterministic == 1)):
        config.deterministic = True
        #config.model.name = config.model.name
        config.model.name = 'det_'+config.model.name
    else:
        config.deterministic = False
    
    if is_main_process():
        print(" >> INSIDE train_model.py: got run_config")
        print(" >> INSIDE train_model.py folder:", str(os.path.join(os.getenv('DERIVED_DATA'), config.data.dataset_name, config.experiment_name)))
    logger.info(" >> INSIDE train_model.py: got run_config")
    logger.info(" >> INSIDE train_model.py folder: %s", str(os.path.join(os.getenv('DERIVED_DATA'), config.data.dataset_name, config.experiment_name)))

    logger.info(" >> INSIDE train_model.py config.deterministic %s, sde: %s", config.deterministic, config.training.sde)

    target_xfm_keys = defaultdict(lambda: config.data.target_transform_key) | dict(config.data.target_transform_overrides)

    # if not 'per_var' in LightningDataModule.__module__:
    #     data_module = LightningDataModule(
    #         active_dataset_name=config.data.dataset_name,
    #         model_src_dataset_name=config.data.dataset_name,
    #         input_transform_dataset_name=config.data.dataset_name,
    #         input_transform_key=config.data.input_transform_key,
    #         target_transform_key=config.data.target_transform_key,
    #         transform_dir=os.path.join(workdir, 'transforms'),
    #         batch_size=config.training.batch_size,
    #         filename=filename,
    #         val_filename=val_filename,
    #         include_time_inputs=False,
    #         evaluation=False,
    #         shuffle=True,
    #         num_workers=3,
    #         prefetch_factor=3
    #     )
    # else:
    data_module = LightningDataModule(
        config=config,
        active_dataset_name=config.data.dataset_name,
        model_src_dataset_name=config.data.dataset_name,
        input_transform_dataset_name=config.data.dataset_name,
        transform_dir=os.path.join(workdir, 'transforms'),
        batch_size=config.training.batch_size,
        filename=filename,
        val_filename=val_filename,
        include_time_inputs=False,
        evaluation=False,
        shuffle=True,
        num_workers=3,
        prefetch_factor=3
    )

    if config.training.random_crop_size > 0:
        random_crop = torchvision.transforms.RandomCrop(config.training.random_crop_size)

    model = ScoreModelLightningModule(config)

    pbar = LossOnlyProgressBar()

    checkpoint_cb, checkpoint_path = setup_checkpoint(config, workdir)

    save_config(config, os.path.join(checkpoint_path, "config.yml"))

    resume_chekpoint_path = check_saved_checkpoint(checkpoint_path)

    trainer = Trainer(
        default_root_dir=os.path.join("lightning_logs", config.data.dataset_name),
        max_epochs=config.training.n_epochs,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
        use_distributed_sampler=True,
        log_every_n_steps=10,
        val_check_interval=1.0, # Run validation at the end of every epoch
        callbacks=[pbar, checkpoint_cb]
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_chekpoint_path
    )