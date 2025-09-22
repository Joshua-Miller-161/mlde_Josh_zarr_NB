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
#===================================================================
FLAGS = flags.FLAGS
#===================================================================
log_dir = os.path.join(os.getcwd(), "Outputs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.log")
open(log_file, 'w').close()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Overwrite mode
        logging.StreamHandler(),  # Also logs to stdout
    ],
)
# Create logger
logger = logging.getLogger(__name__)
logger.info(" << <<< <<<< Logging setup complete. See %s >>>> >>> >>>", log_file)
print(" << <<< <<<< Logging setup complete. See", log_file, " >>>> >>> >>>")
#====================================================================
from src import cncsnpp
from src.lightningModuleEMA import ScoreModelLightningModule
from src.utils import LossOnlyProgressBar, setup_checkpoint, check_saved_checkpoint, is_main_process, create_model
from src.data_scripts.collate_np.data_module import LightningDataModule
#====================================================================
def train(config, workdir, filename):
    
    load_dotenv()

    # save the config
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, 'w') as f:
        f.write(config.to_yaml())

    if is_main_process():
        print(" >> INSIDE run_lib_L: got config")
    logger.info(" >> INSIDE run_lib_L: got config")

    # Create transform saving directory
    transform_dir = os.path.join(workdir, "transforms")
    os.makedirs(transform_dir, exist_ok=True)
    if is_main_process():
        print(" >> INSIDE run_lib_L: got transform_dir")
    logger.info(" >> INSIDE run_lib_L: got transform_dir")

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    if is_main_process():
        print(" >> INSIDE run_lib_L: got transform_dir")
    logger.info(" >> INSIDE run_lib_L: got transform_dir")

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    if is_main_process():
        print(" >> INSIDE run_lib_L: got tb_dir")
    logger.info(" >> INSIDE run_lib_L: got tb_dir")

    target_xfm_keys = defaultdict(lambda: config.data.target_transform_key) | dict(config.data.target_transform_overrides)

    if is_main_process():
        print(" >> INSIDE run_lib_L: got run_config")
        print(" >> INSIDE run_lib_L folder:", os.path.join(os.getenv('DERIVED_DATA'), config.data.dataset_name))
    logger.info(" >> INSIDE run_lib_L: got run_config")
    logger.info(" >> INSIDE run_lib_L folder: %s", os.path.join(os.getenv('DERIVED_DATA'), config.data.dataset_name))

    data_module = LightningDataModule(
        active_dataset_name=config.data.dataset_name,
        model_src_dataset_name=config.data.dataset_name,
        input_transform_dataset_name=config.data.dataset_name,
        input_transform_key=config.data.input_transform_key,
        target_transform_key=config.data.target_transform_key, # Orig, target_transform_keys = target_xfm_keys
        transform_dir=os.path.join(workdir, 'transforms'),
        batch_size=config.training.batch_size,
        filename=filename,
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

    resume_chekpoint_path = check_saved_checkpoint(checkpoint_path)

    trainer = Trainer(
        default_root_dir = os.path.join("lightning_logs", config.data.dataset_name),
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