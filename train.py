import pandas as pd
import torch
import os
import neptune
import logging
import warnings
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.lstm import OgLSTM, CustomLSTM, SeqLSTM
from model.transformer import Transformer   
from model.model_comp import IndoorLocModel
from dataset.dataset import IndoorDataModule
from config import Config
from icecream import ic

def init_config(seed=42):
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    seed_everything(seed)

def init_neptune():
    neptune.init(
        project_qualified_name=,
        api_token=,
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


def load_data():
    # Load data
    train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.csv')
    test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.csv')

    train_data = pd.read_csv(train_data_dir)
    test_data = pd.read_csv(test_data_dir)

    # Init datamodule
    idm = IndoorDataModule(train_data, test_data, kfold=True)
    idm.prepare_data()
    ic(idm.wifi_bssids_size)
    ic(idm.site_id_dim)
    return idm


def train_model(idm: IndoorDataModule, fold: int):
    # Set fold
    ic(fold)
    idm.set_fold_num(fold)
    idm.setup()
    
    # Init model
    model = IndoorLocModel(SeqLSTM(
        Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))

    # Init callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(Config.SAVE_DIR, f'{fold}'),
        filename='{epoch:02d}-{val_loss:.2f}-{val_metric:.2f}.pth',
        save_top_k=5,
        mode='min',
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
    )

    # Init trainer
    trainer = Trainer(
        gpus=1,
        num_sanity_val_steps=-1,
        deterministic=True,
        max_epochs=Config.epochs,
        callbacks=[checkpoint_callback, early_stopping],
        # profiler="simple",
    )
    # trainer.tune(model, idm)

    # Train
    trainer.fit(model, idm)


def main():
    init_config()

    if Config.neptune:
        init_neptune()

    idm = load_data()

    for fold in range(Config.fold_num):
        train_model(idm, fold)


if __name__ == "__main__":
    main()
