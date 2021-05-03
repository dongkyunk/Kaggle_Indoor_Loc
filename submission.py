import pandas as pd
import torch
import os
import neptune
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.lstm import OgLSTM, CustomLSTM
from model.model_comp import IndoorLocModel
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
idm.setup(stage="test")
ic(idm.wifi_bssids_size)
ic(idm.site_id_dim)

model_path_0 = os.path.join(Config.SAVE_DIR, '0/epoch=158-val_loss=4.99-val_metric=4.99.pth.ckpt')
model_path_1 = os.path.join(Config.SAVE_DIR, '1/epoch=153-val_loss=5.30-val_metric=5.30.pth.ckpt')
model_path_2 = os.path.join(Config.SAVE_DIR, '2/epoch=123-val_loss=4.99-val_metric=4.99.pth.ckpt')
model_path_3 = os.path.join(Config.SAVE_DIR, '3/epoch=121-val_loss=5.22-val_metric=5.22.pth.ckpt')
model_path_4 = os.path.join(Config.SAVE_DIR, '4/epoch=129-val_loss=5.14-val_metric=5.14.pth.ckpt')


model0 = IndoorLocModel.load_from_checkpoint(model_path_0, model=CustomLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
model0.eval()
model1 = IndoorLocModel.load_from_checkpoint(model_path_1, model=CustomLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
model1.eval()
model2 = IndoorLocModel.load_from_checkpoint(model_path_2, model=CustomLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
model2.eval()
model3 = IndoorLocModel.load_from_checkpoint(model_path_3, model=CustomLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
model3.eval()
model4 = IndoorLocModel.load_from_checkpoint(model_path_4, model=CustomLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
model4.eval()


for i, batch in enumerate(idm.test_dataloader()):
    batch_index = i * Config.val_batch_size
    
    # Make prediction
    output = torch.cat([model0(batch).unsqueeze(1), 
                        model1(batch).unsqueeze(1),
                        model2(batch).unsqueeze(1),
                        model3(batch).unsqueeze(1),], dim=1)
    output = torch.mean(output, 1, keepdim=True).squeeze()
    # output = model0(batch)
    x = output[:, 0].cpu().detach().numpy()
    y = output[:, 1].cpu().detach().numpy()
    f = output[:, 2].cpu().detach().numpy()

    submit.iloc[batch_index:batch_index+Config.val_batch_size, -3] = f
    submit.iloc[batch_index:batch_index+Config.val_batch_size, -2] = x
    submit.iloc[batch_index:batch_index+Config.val_batch_size, -1] = y

ic(submit)
b = pd.read_csv("data/submission-6.csv")
submit['floor'] = b['floor']
submit.to_csv('data/submit.csv', index=False)