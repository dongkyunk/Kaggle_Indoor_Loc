import pandas as pd
import torch
import os
import neptune
from pytorch_lightning import Trainer
from model.lstm import simpleLSTM
from model.model import IndoorLocModel
from dataset.dataset import IndoorDataModule
from config import Config
from icecream import ic


# neptune.init(project_qualified_name='dongkyuk/IndoorLoc',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWY4YTFhZS00NGU5LTQxOTUtOGI5NC04ZjgwOTJkMDFmNjYifQ==',
#              )
# neptune.create_experiment()


def train_model(fold: int):
    train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.pkl')
    test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.pkl')

    train_data = pd.read_pickle(train_data_dir)
    test_data = pd.read_pickle(test_data_dir)

    idm = IndoorDataModule(train_data, test_data, kfold=True, fold_num=fold)
    idm.prepare_data()
    idm.setup()
    model = IndoorLocModel(simpleLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))

    trainer = Trainer()
    trainer.fit(model, idm)


def main():
    for fold in range(Config.fold_num):
        train_model(fold)

if __name__ == "__main__":
    main()
