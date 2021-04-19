import pandas as pd
import torch
import os
import neptune
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.lstm import OrgLSTM
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

idm = IndoorDataModule(train_data, test_data, kfold=True, fold_num=0)
idm.prepare_data()
idm.setup()

model_path = os.path.join(Config.SAVE_DIR, '0/epoch=90-val_loss=7772.81.pth.ckpt')
model = IndoorLocModel.load_from_checkpoint(model_path, model=OrgLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
trainer = Trainer(
    gpus=1
)


for i, batch in enumerate(idm.test_dataloader()):
    batch_index = i * Config.batch_size
    
    # Make prediction
    xy_hat, floor_hat = model(batch)
    x = xy_hat[:, 0].cpu().detach().numpy()
    y = xy_hat[:, 1].cpu().detach().numpy()
    f = torch.argmax(floor_hat, dim=-1).cpu().detach().numpy() - 2

    submit.iloc[batch_index:batch_index+Config.batch_size, -3] = f
    submit.iloc[batch_index:batch_index+Config.batch_size, -2] = x
    submit.iloc[batch_index:batch_index+Config.batch_size, -1] = y

submit.to_csv('data/submit.csv', index=False)