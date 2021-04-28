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
idm.setup(stage="test")
ic(idm.wifi_bssids_size)
ic(idm.site_id_dim)

# model_path_0 = os.path.join(Config.SAVE_DIR, '0/epoch=32-val_loss=6.16.pth.ckpt')
# model_path_1 = os.path.join(Config.SAVE_DIR, '1/epoch=35-val_loss=6.13.pth.ckpt')
# model_path_2 = os.path.join(Config.SAVE_DIR, '2/epoch=39-val_loss=5.99.pth.ckpt')
# model_path_3 = os.path.join(Config.SAVE_DIR, '3/epoch=38-val_loss=6.19.pth.ckpt')
model_path_4 = os.path.join(Config.SAVE_DIR, '0/epoch=10-val_loss=116.67-val_metric=0.00.pth.ckpt')


# model0 = IndoorLocModel.load_from_checkpoint(model_path_0, model=OgLSTM(
#         Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
# model1 = IndoorLocModel.load_from_checkpoint(model_path_1, model=OgLSTM(
#         Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
# model2 = IndoorLocModel.load_from_checkpoint(model_path_2, model=OgLSTM(
#         Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
# model3 = IndoorLocModel.load_from_checkpoint(model_path_3, model=OgLSTM(
#         Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
model4 = IndoorLocModel.load_from_checkpoint(model_path_4, model=OgLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
model4.eval()

# def xy_loss(xy_hat, xy_label):
#     xy_loss = torch.mean(torch.sqrt(
#         (xy_hat[:, 0]-xy_label[:, 0])**2 + (xy_hat[:, 1]-xy_label[:, 1])**2))
#     return xy_loss


# def floor_loss(floor_hat, floor_label):
#     floor_loss = 15 * torch.mean(torch.abs(floor_hat-floor_label))
#     return floor_loss

# batch = next(iter(idm.train_dataloader()))
# # batch = next(iter(idm.train_dataloader()))
# # batch = next(iter(idm.train_dataloader()))

# x, y, f = batch['x'], batch['y'], batch['floor']

# xy_label = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

# output = model4(batch)

# xy_hat = output[:, 0:2]
# f_hat = output[:, 2]
# ic(xy_hat)
# ic(xy_hat[:2])
# ic(xy_label)
# ic(f)
# ic(f_hat)
# loss_xy = xy_loss(xy_hat[:2], xy_label[:2])
# loss_floor = floor_loss(f_hat[:2], f[:2])
# ic(loss_xy)
# ic(loss_floor)
# loss_xy = xy_loss(xy_hat[2:4], xy_label[2:4])
# loss_floor = floor_loss(f_hat[2:4], f[2:4])
# ic(loss_xy)
# ic(loss_floor)
# loss_xy = xy_loss(xy_hat, xy_label)
# loss_floor = floor_loss(f_hat, f)
# ic(loss_xy)
# ic(loss_floor)


# trainer = Trainer(
#     gpus=1
# )
# ic(idm.train_data.head())
# ic(idm.test_data.head())
# ic(idm.train_data['site_id'].head())
# ic(idm.test_data['site_id'].head())

for i, batch in enumerate(idm.test_dataloader()):
    batch_index = i * Config.val_batch_size
    
    # Make prediction
#     output = torch.cat([model0(batch).unsqueeze(1), 
#                         model1(batch).unsqueeze(1),
#                         model2(batch).unsqueeze(1),
#                         model3(batch).unsqueeze(1),], dim=1)
#     output = torch.mean(output, 1, keepdim=True).squeeze()
    output = model4(batch)
    x = output[:, 0].cpu().detach().numpy()
    y = output[:, 1].cpu().detach().numpy()
    # f = torch.argmax(floor_hat, dim=-1).cpu().detach().numpy() - 2
    # f = output[:, 2].cpu().detach().numpy()

    # submit.iloc[batch_index:batch_index+Config.val_batch_size, -3] = f
    submit.iloc[batch_index:batch_index+Config.val_batch_size, -2] = x
    submit.iloc[batch_index:batch_index+Config.val_batch_size, -1] = y

ic(submit)

b = pd.read_csv("data/submission-6.csv")
submit['floor'] = b['floor']
submit.to_csv('data/submit.csv', index=False)