#!/usr/bin/env python3
"""
create_transforms_example.py

Example script to build & persist input/target transforms for the
Zarr dataset at /home/dankycush/train.zarr using the provided config.

Assumptions:
- You have the NumPy-based transforms module available (example name: transforms_numpy.py).
- xarray and yaml are installed.
- The Zarr dataset path is /home/dankycush/train.zarr.
"""

import sys
sys.dont_write_bytecode=True
import logging
from pathlib import Path
import yaml
import xarray as xr

# Adjust this import to the module name where the NumPy transforms live.
# If you named it differently, update the import line.
import mlde_Josh_zarr_NB.src.transforms_np as tx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create_transforms")

# ========== CONFIGURATION ==========
ZARR_PATH = "/work/scratch-nopw2/j_miller/data/big_file/val_consolodated.zarr"
DS_CONFIG_PATH = "/work/scratch-nopw2/j_miller/data/zarr/ds-config.yml"  # path to the YAML you provided
TRANSFORM_BASE_DIR = "/gws/nopw/j04/bris_climdyn/j_miller/temp/transforms"
ACTIVE_DATASET_NAME = "zarr"    # name used in foldering (matches your default config data.dataset_name)
MODEL_SRC_DATASET_NAME = "zarr" # we use the same dataset as the model source here
INPUT_KEY = "stan"                   # from default_config.py: data.input_transform_key
TARGET_KEY = "sqrturrecen"           # from default_config.py: data.target_transform_key
# ===================================

def read_ds_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    logger.info("Reading dataset config: %s", DS_CONFIG_PATH)
    ds_cfg = read_ds_config(DS_CONFIG_PATH)

    # Extract variables
    predictors = ds_cfg.get("predictors", {}) or {}
    predictands = ds_cfg.get("predictands", {}) or {}

    variables = predictors.get("variables", [])
    target_variables = predictands.get("variables", [])

    if not variables:
        logger.error("No predictor variables found in %s", DS_CONFIG_PATH)
        sys.exit(1)
    if not target_variables:
        logger.error("No predictand variables found in %s", DS_CONFIG_PATH)
        sys.exit(1)

    logger.info("Predictor variables: %s", variables)
    logger.info("Target variables: %s", target_variables)

    # Build transforms (unfitted)
    logger.info("Building input transform (key=%s)...", INPUT_KEY)
    input_xfm = tx.build_input_transform(variables, INPUT_KEY)

    logger.info("Building target transform (key=%s) for each target variable...", TARGET_KEY)
    # build_target_transform expects a dict keyed by variable of keys
    target_keys_map = {v: TARGET_KEY for v in target_variables}
    target_xfm = tx.build_target_transform(target_variables, target_keys_map)

    # Prepare transform output directory
    dataset_transform_dir = Path(TRANSFORM_BASE_DIR) / ACTIVE_DATASET_NAME / (INPUT_KEY+'-'+TARGET_KEY)
    dataset_transform_dir.mkdir(parents=True, exist_ok=True)
    input_transform_path = dataset_transform_dir / "input.pickle"
    target_transform_path = dataset_transform_dir / "target.pickle"

    # Open the Zarr dataset (xarray)
    logger.info("Opening Zarr dataset: %s", ZARR_PATH)
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)

    try:
        # Convert only needed variables to numpy dicts using helper from transforms module.
        # For the input transform we typically use the predictor variables
        logger.info("Converting predictor variables to numpy arrays and fitting input transform...")
        predictor_np = tx._ensure_numpy_dict(ds, variables)
        # Following the convention in the transforms module: fit(active_ds, model_src_ds)
        # Here we use the same dataset as both active and model source (adapt if different)
        input_xfm.fit(predictor_np, predictor_np)

        logger.info("Saving input transform to %s", input_transform_path)
        tx.save_transform(input_xfm, str(input_transform_path))

        # Fit target transform (use the predictands variables)
        logger.info("Converting target variables to numpy arrays and fitting target transform...")
        target_np = tx._ensure_numpy_dict(ds, target_variables)
        target_xfm.fit(target_np, predictor_np)  # note: model_src argument passed as predictor_np here; adjust if you need different
        logger.info("Saving target transform to %s", target_transform_path)
        tx.save_transform(target_xfm, str(target_transform_path))

    finally:
        # cleanup
        try:
            if hasattr(ds, "close"):
                ds.close()
        except Exception:
            pass

    logger.info("Transforms created and saved successfully.")
    logger.info("Input transform:  %s", input_transform_path)
    logger.info("Target transform: %s", target_transform_path)


if __name__ == "__main__":
    main()