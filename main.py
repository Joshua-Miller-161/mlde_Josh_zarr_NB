# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Modifications copyright 2024 Henry Addison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training"""
import sys
sys.dont_write_bytecode = True
import os
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
#import logging
import os

sys.path.append(os.getcwd())
from src import run_lib_orig as run_lib
#====================================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("dm_type", None, "Which method to load data")
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train"], "Running mode: train")
flags.DEFINE_string("filename", None, "File to train on")
flags.DEFINE_string("val_filename", None, "File containing the validation data")
flags.mark_flags_as_required(["workdir", "config", "mode", "filename", "val_filename"])

def main(argv):
    if FLAGS.mode == "train":
        # Create the working directory
        os.makedirs(FLAGS.workdir, exist_ok=True)

        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir, FLAGS.filename, FLAGS.val_filename)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)