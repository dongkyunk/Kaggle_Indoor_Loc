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
    epochs = 15
    num_wifi_feats = 20
    fold_num = 5
    batch_size = 32

    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    tqdm.pandas()
    seed_everything(seed)
