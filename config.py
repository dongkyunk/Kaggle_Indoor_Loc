import logging
import warnings
import random
import os
import numpy as np
import torch
from modin.config import ProgressBar
from tqdm.auto import tqdm
from pytorch_lightning.utilities.seed import seed_everything

class Config():
    DATA_DIR = 'data'
    SAVE_DIR = 'save'
    
    seed = 42
    epochs = 300
    num_wifi_feats = 20
    fold_num = 5
    train_batch_size = 128
    val_batch_size = 256
    num_workers = 24
    device = 'gpu'
    neptune = False
    lr = 5e-3

    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    tqdm.pandas()
    seed_everything(seed)
