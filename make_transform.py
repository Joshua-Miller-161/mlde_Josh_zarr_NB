import sys
sys.dont_write_bytecode=True
import os
import logging
from pathlib import Path
import yaml
import xarray as xr
import zarr
from dotenv import load_dotenv
import time as clock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.cm import get_cmap

from src.transforms_np import _build_transform_per_variable_from_config, _find_or_create_transforms_per_variable_from_config
from configs.subvpsde.ukcp_local_pr_1em_cncsnpp_continuous import get_config
#====================================================================
print(zarr.__version__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create_transforms")

# =========================== CONFIGURATION =========================
ZARR_PATH = "/work/scratch-nopw2/j_miller/data/zarr/train_consolidated_time1_elev.zarr"
DS_CONFIG_PATH = "/work/scratch-nopw2/j_miller/data/zarr/ds-config.yml"  # path to the YAML you provided
TRANSFORM_BASE_DIR = "/gws/nopw/j04/bris_climdyn/j_miller/temp/transforms"
ACTIVE_DATASET_NAME = "zarr"    # name used in foldering (matches your default config data.dataset_name)
MODEL_SRC_DATASET_NAME = "zarr" # we use the same dataset as the model source here
#====================================================================
def main():
    load_dotenv()

    config = get_config()

    # Extract variables
    variables = config.data.predictors.variables #predictors.get("variables", [])
    target_variables = config.data.predictands.variables #predictands.get("variables", [])
    
    logger.info("Predictor variables: %s", variables)
    logger.info("Target variables: %s", target_variables)

    input_transforms, target_transforms = _find_or_create_transforms_per_variable_from_config(
        ZARR_PATH,
        ACTIVE_DATASET_NAME,
        MODEL_SRC_DATASET_NAME,
        TRANSFORM_BASE_DIR,
        config,
        False
    )

if __name__ == "__main__":
    main()