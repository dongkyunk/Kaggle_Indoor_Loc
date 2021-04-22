import pandas as pd
# import torch
# import os

# from model.lstm import simpleLSTM
# from dataset.dataset import IndoorDataModule
# from config import Config
from icecream import ic

# train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.pkl')
# test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.pkl')

# train_data = pd.read_pickle(train_data_dir)
# test_data = pd.read_pickle(test_data_dir)

# idm = IndoorDataModule(train_data, test_data, kfold=True, fold_num=0)
# idm.prepare_data()
# idm.setup()
# sample_batch = next(iter(idm.train_dataloader()))

# # model = simpleLSTM()
# slstm = simpleLSTM(20, 61206, 1000)
# slstm.forward(sample_batch)
# # trainer = Trainer()
# # trainer.fit(model, idm)

a = pd.read_csv("data/submit.csv")
ic(a.head())

b = pd.read_csv("data/submission-6.csv")

a['floor'] = b['floor']
a.to_csv("data/submit.csv", index=False)