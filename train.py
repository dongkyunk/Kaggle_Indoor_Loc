import pandas as pd
import torch
import os

from model.
from dataset.dataset import IndoorDataModule
from config import Config

train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.pkl') 
test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.pkl') 

idm = IndoorDataModule(train_data_dir, test_data_dir, Config.batch_size, cross_val=True, Config.fold_num)
model = LitClassifier()

trainer = Trainer()
trainer.fit(model, idm)
