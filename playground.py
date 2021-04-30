import pandas as pd
import torch
import os
import neptune
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.lstm import OgLSTM
from dataset.dataset import IndoorDataModule
from config import Config
from icecream import ic

# a = pd.read_csv("data/submission-11.csv")
# b = pd.read_csv("data/sample_submission.csv")

# a.index = a['site_path_timestamp']
# ic(a)
# a = a.reindex(b['site_path_timestamp'])
# ic(a)
# a.reset_index(drop=True, inplace=True) 
# ic(a)
# a.to_csv("data/submit.csv", index=False)

import torch.nn as nn
a =  nn.Parameter(torch.randn(1, 1, 2))
ic(a)
a = a.repeat(4, 1, 1)
ic(a)
ic(a.shape)