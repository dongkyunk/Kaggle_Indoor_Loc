import pandas as pd
from icecream import ic
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from config import Config
from utils.utils import time_function
import numpy as np


class IndoorDataset(Dataset):
    def __init__(self, data, bssid_feats, rssi_feats, flag='TRAIN'):
        self.data = data
        self.flag = flag
        self.bssid_feats = bssid_feats
        self.rssi_feats = rssi_feats

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        tmp_data = self.data.iloc[index]
        if self.flag == 'TRAIN':
            return {
                'BSSID_FEATS': tmp_data[self.bssid_feats].values.astype(int),
                'RSSI_FEATS': tmp_data[self.rssi_feats].values.astype(np.float32),
                'site_id': tmp_data['site_id'].astype(int),
                'x': tmp_data['x'],
                'y': tmp_data['y'],
                'floor': tmp_data['floor'],
            }
        elif self.flag == 'TEST':
            return {
                'BSSID_FEATS': tmp_data[self.bssid_feats].values.astype(int),
                'RSSI_FEATS': tmp_data[self.rssi_feats].values.astype(np.float32),
                'site_id': tmp_data['site_id'].astype(int)
            }


class IndoorDataModule(LightningDataModule):
    def __init__(self, train_data, test_data, kfold=False):
        self.train_data = train_data
        self.test_data = test_data
        self.kfold = kfold

    def set_fold_num(self, fold_num):
        self.fold_num = fold_num

    def _init_feats(self):
        self.bssid_feats = [f'bssid_{i}' for i in range(Config.num_wifi_feats)]
        self.rssi_feats = [f'rssi_{i}' for i in range(Config.num_wifi_feats)]

    def _init_wifi_bssids(self):
        wifi_bssids = []
        for i in range(100):
            wifi_bssids += self.train_data[f'bssid_{i}'].values.tolist()
            wifi_bssids += self.test_data[f'bssid_{i}'].values.tolist()

        self.wifi_bssids = list(set(wifi_bssids))
        self.wifi_bssids_size = len(self.wifi_bssids)

    def _init_transforms(self):
        self.wifi_bssids_encoder = LabelEncoder()
        self.wifi_bssids_encoder.fit(self.wifi_bssids)

        self.site_id_encoder = LabelEncoder()
        self.site_id_encoder = self.site_id_encoder.fit(
            self.train_data['site_id'])

        self.rssi_normalizer = StandardScaler()
        self.rssi_normalizer.fit(self.train_data[self.rssi_feats])

    def _transform(self, data):
        for bssid_feat in self.bssid_feats:
            data[bssid_feat] = self.wifi_bssids_encoder.transform(
                data[bssid_feat])
        data['site_id'] = self.site_id_encoder.transform(data['site_id'])
        data[self.rssi_feats] = self.rssi_normalizer.transform(
            data[self.rssi_feats])
        return data

    def _kfold(self):
        ''' Group Kfold wrt path and Stratified Kfold wrt site_id
        '''
        skf = StratifiedKFold(n_splits=Config.fold_num,
                                   shuffle=True, random_state=Config.seed)
        self.train_data['site_id_f'] = self.train_data['site_id'] + self.train_data['floor'].astype(str)
        for n, (train_index, val_index) in enumerate(
            skf.split(
                X = self.train_data['path'],
                y = self.train_data['path']
            )
        ):
            self.train_data.loc[val_index, 'kfold'] = int(n)

    @time_function
    def prepare_data(self):
        # Init cross validation
        if self.kfold:
            self._kfold()

        # Init preprocessing
        self._init_feats()
        self._init_wifi_bssids()
        self._init_transforms()
        self.site_id_dim = len(self.train_data['site_id'].unique())
        self.train_data = self._transform(self.train_data)
        self.test_data = self._transform(self.test_data)

    @time_function
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.kfold:
                train_df = self.train_data[self.train_data['kfold'] !=
                                           self.fold_num].reset_index(drop=True)
                val_df = self.train_data[self.train_data['kfold'] ==
                                         self.fold_num].reset_index(drop=True)
            self.train = IndoorDataset(
                train_df, self.bssid_feats, self.rssi_feats, flag="TRAIN")
            self.val = IndoorDataset(
                val_df, self.bssid_feats, self.rssi_feats, flag="TRAIN")

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = IndoorDataset(
                self.test_data, self.bssid_feats, self.rssi_feats, flag="TEST")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=Config.train_batch_size, num_workers=Config.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=Config.val_batch_size, num_workers=Config.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=Config.val_batch_size, num_workers=Config.num_workers, shuffle=False, pin_memory=True)
