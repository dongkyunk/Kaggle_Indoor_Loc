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
                'BSSID_FEATS':tmp_data[BSSID_FEATS].values.astype(float),
                'RSSI_FEATS':tmp_data[RSSI_FEATS].values.astype(float),
                'site_id':tmp_data['site_id'].astype(int),
                'x':tmp_data['x'],
                'y':tmp_data['y'],
                'floor':tmp_data['floor'],
            }
        elif self.:
            return {
                'BSSID_FEATS':tmp_data[BSSID_FEATS].values.astype(float),
                'RSSI_FEATS':tmp_data[RSSI_FEATS].values.astype(float),
                'site_id':tmp_data['site_id'].astype(int)