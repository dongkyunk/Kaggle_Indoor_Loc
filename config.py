import logging
import warnings
import random
import os
import numpy as np
import torch
from modin.config import ProgressBar
from tqdm.auto import tqdm

class Config():
    DATA_DIR = 'data'
    SAVE_DIR = 'save'
    
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    tqdm.pandas()

    seed = 42
    epochs = 15
    num_wifi_feats = 20