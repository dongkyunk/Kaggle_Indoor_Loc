import pandas as pd
import torch
import os

from dataset.dataset import IndoorDataModule
from config import Config

train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.pkl') 
test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.pkl') 

mnist = IndoorDataModule(train_data_dir, test_data_dir)
