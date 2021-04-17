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

if Config.neptune:
    neptune.init(project_qualified_name='dongkyuk/IndoorLoc',
                api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWY4YTFhZS00NGU5LTQxOTUtOGI5NC04ZjgwOTJkMDFmNjYifQ==',
                )
    neptune.create_experiment(
        upload_source_files=[
            "train.py",
            "model/lstm.py",
            "model/model.py",
            "dataset/dataset.py",
            "config.py",
        ]
    )


def train_model(fold: int):
    train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.pkl')
    test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.pkl')

    train_data = pd.read_pickle(train_data_dir)
    test_data = pd.read_pickle(test_data_dir)

    idm = IndoorDataModule(train_data, test_data, kfold=True, fold_num=fold)
    idm.prepare_data()
    idm.setup()
    ic(idm.wifi_bssids_size)
    ic(idm.site_id_dim)
    model = IndoorLocModel(OrgLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(Config.SAVE_DIR, f'{fold}'),
        filename='{epoch:02d}-{val_loss:.2f}.pth',
        save_top_k=5,
        mode='min',
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
    )

    trainer = Trainer(
        gpus=1, 
        num_sanity_val_steps=-1,
        deterministic=True, 
        max_epochs=Config.epochs, 
        callbacks=[checkpoint_callback, early_stopping],
        auto_lr_find=True
    )
    # trainer.tune(model, idm)
    trainer.fit(model, idm)


def main():
    for fold in range(Config.fold_num):
        train_model(fold)

if __name__ == "__main__":
    main()
