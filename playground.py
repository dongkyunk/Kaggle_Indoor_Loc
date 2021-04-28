import pandas as pd
import torch
import os
import neptune
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.lstm import OgLSTM
from model.model import IndoorLocModel
from dataset.dataset import IndoorDataModule
from config import Config
from icecream import ic

train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.pkl')
test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.pkl')
submit_dir = os.path.join(Config.DATA_DIR, 'sample_submission.csv')

train_data = pd.read_pickle(train_data_dir)
test_data = pd.read_pickle(test_data_dir)
submit = pd.read_csv(submit_dir)

idm = IndoorDataModule(train_data, test_data, kfold=False)
idm.prepare_data()
idm.setup()
ic(idm.wifi_bssids_size)
ic(idm.site_id_dim)

ic(idm.test_data.head)
ic(idm.train_data)