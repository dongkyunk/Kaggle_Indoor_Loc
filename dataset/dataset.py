class IndoorDataset(Dataset):
    def __init__(self, data, flag='TRAIN'):
        self.data = data
        self.flag = flag

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        tmp_data = self.data.iloc[index]
        if self.flag == 'TRAIN':
            return {
                'BSSID_FEATS': tmp_data[BSSID_FEATS].values.astype(float),
                'RSSI_FEATS': tmp_data[RSSI_FEATS].values.astype(float),
                'site_id': tmp_data['site_id'].astype(int),
                'x': tmp_data['x'],
                'y': tmp_data['y'],
                'floor': tmp_data['floor'],
            }
        elif self.flag == 'TEST':
            return {
                'BSSID_FEATS': tmp_data[BSSID_FEATS].values.astype(float),
                'RSSI_FEATS': tmp_data[RSSI_FEATS].values.astype(float),
                'site_id': tmp_data['site_id'].astype(int)
            }


class IndoorDataModule():
    def __init__(self, train_dir, test_dir, batch_size, cross_val=False, fold_num=None):
        self.train_data = pd.read_pickle(train_dir)
        self.test_data = pd.read_pickle(test_dir)
        self.batch_size = batch_size

        # Init cross validation
        self.cross_val = cross_val
        self.fold_num = fold_num
        if self.cross_val:
            self._kfold()

        # Init preprocessing
        self._init_wifi_bssids()
        self._init_transforms()

    def _init_wifi_bssids(self):
        wifi_bssids = []
        for i in range(100):
            wifi_bssids.extend(self.train_data.iloc[:, i].values.tolist())
            wifi_bssids.extend(self.test_data.iloc[:, i].values.tolist())
        self.wifi_bssids = list(set(wifi_bssids))
        self.wifi_bssids_size = len(wifi_bssids)

    def _init_transforms(self):
        self.wifi_bssids_encoder = LabelEncoder()
        self.wifi_bssids_encoder.fit(self.wifi_bssids)

        self.site_id_encoder = LabelEncoder()
        self.site_id_encoder.fit(self.train_data['site_id'])

        self.rssi_normalizer = StandardScaler()
        self.rssi_normalizer.fit(self.train_data.loc[:, RSSI_FEATS])

    def _transform(self, data):
        data.loc[:, BSSI_FEATS] = self.wifi_bssids_encoder.transform(
            data.loc[:, BSSI_FEATS])
        data.loc[:, 'site_id'] = self.site_id_encoder.transform(
            data.loc[:, 'site_id'])
        data.loc[:, RSSI_FEATS] = self.rssi_normalizer.transform(
            data.loc[:, RSSI_FEATS])
        return data

    def _kfold(self):
        ''' Stratified Kfold based on site_id 
        '''
        self.other, self.site_id = self.train_data.drop(
            columns=['site_id']), self.train_data['site_id']
        skf = StratifiedKFold(n_splits=Config.FOLD_NUM,
                              shuffle=True, random_state=Config.SEED)
        for n, (train_index, val_index) in enumerate(skf.split(self.other, self.site_id)):
            self.train_data.loc[val_index, 'fold'] = int(n)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_data = self._transform(self.train_data)
            if self.cross_val:
                train_df = self.df[self.df['kfold'] !=
                                   self.fold_num].reset_index(drop=True)
                val_df = self.df[self.df['kfold'] ==
                                 self.fold_num].reset_index(drop=True)
            self.train = IndoorDataset(train_df, flag="TRAIN")
            self.train = IndoorDataset(val_df, flag="TRAIN")

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = self._transform(self.test_data)
            self.test = IndoorDataset(self.test_data, flag="TEST")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
