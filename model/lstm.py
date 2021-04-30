import numpy as np
import torch
import torch.nn as nn
from icecream import ic


class OgLSTM(nn.Module):
    def __init__(self, input_dim, bssid_dim, site_id_dim, embedding_dim=64, seq_len=20):
        super(OgLSTM, self).__init__()

        self.feature_dim = input_dim * embedding_dim * 2 + 2

        # Embedding
        self.embd_bssid = nn.Embedding(bssid_dim, embedding_dim)
        self.embd_site_id = nn.Embedding(site_id_dim, 2)

        # LSTM
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128,
                             dropout=0.3, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=16,
                             dropout=0.1, bidirectional=False)

        self.fc_rssi = nn.Linear(input_dim, input_dim * embedding_dim)
        self.fc_features = nn.Linear(self.feature_dim, 256)
        self.fc_output = nn.Linear(16, 3)

        self.batch_norm_rssi = nn.BatchNorm1d(input_dim)
        self.batch_norm1 = nn.BatchNorm1d(self.feature_dim)
        self.batch_norm2 = nn.BatchNorm1d(1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embd_bssid = self.embd_bssid(x['BSSID_FEATS'])
        embd_bssid = torch.flatten(embd_bssid, start_dim=-2)

        embd_site_id = self.embd_site_id(x['site_id'])  # (,) -> (,2)
        embd_site_id = torch.flatten(embd_site_id, start_dim=-1)  # (,2)

        rssi_feat = self.batch_norm_rssi(x['RSSI_FEATS'])  # (,input_dim)
        rssi_feat = self.fc_rssi(rssi_feat)  # (, input_dim * embedding_dim)
        rssi_feat = torch.relu(rssi_feat)  # (, input_dim * embedding_dim)

        x = torch.cat([embd_bssid, embd_site_id, rssi_feat], dim=-1)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.fc_features(x)
        x = torch.relu(x)  # (, 256)

        x = x.unsqueeze(-2)  # (, 1, 256)
        x = self.batch_norm2(x)  # (, 1, 256)
        x = x.transpose(0, 1)  # (1, , 256)
        x, _ = self.lstm1(x)  # (1, , 128)
        x = x.transpose(0, 1)  # (, 1, 128)

        x = torch.relu(x)
        x = x.transpose(0, 1)
        x, _ = self.lstm2(x)  # (1, , 16)
        x = x.transpose(0, 1)  # (, 1, 16)
        x = torch.relu(x)

        output = self.fc_output(x).squeeze()

        return output


class CustomLSTM(nn.Module):
    def __init__(self, wifi_num, bssid_dim, site_id_dim, embedding_dim=32):
        """CustomLSTM Model

        Args:
            wifi_num (int): number of wifi signals to use
            bssid_dim (int): total number of unique bssids
            site_id_dim (int): total number of unique site ids
            embedding_dim (int): Dimension of bssid embedding. Defaults to 64.
        """
        super(CustomLSTM, self).__init__()
        self.wifi_num = wifi_num
        self.feature_dim = 256

        # Embedding
        self.embd_bssid = nn.Embedding(bssid_dim, embedding_dim)
        self.embd_site_id = nn.Embedding(site_id_dim, 2)

        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128,
                             dropout=0.3, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=16,
                             dropout=0.1, bidirectional=False)

        # Linear
        self.fc_rssi = nn.Linear(wifi_num, wifi_num * embedding_dim)
        self.fc_features = nn.Linear(
            wifi_num * embedding_dim * 2 + 2, self.feature_dim)
        self.fc_output = nn.Linear(16, 3)

        # Other
        self.bn_rssi = nn.BatchNorm1d(wifi_num * embedding_dim)
        self.bn_features = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embd_bssid = self.embd_bssid(x['BSSID_FEATS'])  # (,wifi_num,64)
        embd_bssid = torch.flatten(
            embd_bssid, start_dim=-2)  # (,wifi_num * 64)

        embd_site_id = self.embd_site_id(x['site_id'])  # (,2)
        embd_site_id = torch.flatten(embd_site_id, start_dim=-1)

        rssi_feat = x['RSSI_FEATS']          # (,wifi_num)
        rssi_feat = self.fc_rssi(rssi_feat)  # (,64)
        rssi_feat = self.bn_rssi(rssi_feat)
        rssi_feat = torch.relu(rssi_feat)

        x = torch.cat([embd_bssid, embd_site_id, rssi_feat], dim=-1)
        x = self.fc_features(x)
        x = self.bn_features(x)
        x = self.dropout(x)
        x = torch.relu(x)  # (, 256)

        x = x.unsqueeze(-2)  # (, 1, 256)
        x = x.transpose(0, 1)  # (1, , 256)
        x, _ = self.lstm1(x)  # (1, , 128)
        x = x.transpose(0, 1)  # (, 1, 128)
        x = torch.relu(x)

        x = x.transpose(0, 1)
        x, _ = self.lstm2(x)  # (1, , 16)
        x = x.transpose(0, 1)  # (, 1, 16)
        x = torch.relu(x)

        output = self.fc_output(x).squeeze()

        return output
