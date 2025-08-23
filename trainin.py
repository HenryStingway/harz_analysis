from deepforest import main
from deepforest import get_data
from deepforest import utilities
from deepforest import preprocess

import os
import time
import numpy as np

# Example run with short training
train_file = get_data("neon_tree_dataset/train/annotations.csv")
val_file = get_data('neon_tree_dataset/val/annotations.csv')

#initial the model and change the corresponding config file
m = main.deepforest()
m.config["save-snapshot"] = False
m.config['gpus'] = '1' #move to GPU and use all the GPU resources
m.config["train"]["csv_file"] = train_file
m.config["train"]["root_dir"] = 'neon_tree_dataset/train/img'
m.config["score_thresh"] = 0.4
m.config["train"]['epochs'] = 25
m.config["validation"]["csv_file"] = val_file
m.config["validation"]["root_dir"] = 'neon_tree_dataset/val/img'
#create a pytorch lighting trainer used to training

m.create_trainer()

start_time = time.time()
m.trainer.fit(m)

m.trainer.save_checkpoint("weights/checkpoint.pl")
print(f"--- Training on CPU: {(time.time() - start_time):.2f} seconds ---")