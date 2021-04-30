import numpy as np
import torch
import torch.nn as nn
from icecream import ic


class Transformer(nn.Module):
    def __init__(self, wifi_num, bssid_dim, site_id_dim, embedding_dim=32):
        """Transformer Model

        Args:
            wifi_num (int): number of wifi signals to use
            bssid_dim (int): total number of unique bssids
            site_id_dim (int): total number of unique site ids
            embedding_dim (int): Dimension of bssid embedding. Defaults to 64.
        """
        super(Transformer, self).__init__()
        self.wifi_num = wifi_num
        self.feature_dim = 64

        # Embedding
        self.embd_bssid = nn.Embedding(bssid_dim, embedding_dim)
        self.embd_site_id = nn.Embedding(site_id_dim, 2)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, dropout=0.1, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=6)
        self.reg_token = nn.Parameter(torch.randn(1, 1, embedding_dim * 2 + 2))

        # Linear
        self.fc_rssi = nn.Linear(1, embedding_dim)
        self.fc_features = nn.Linear(embedding_dim * 2 + 2, self.feature_dim)
        self.fc_output = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 3),
        )

        # Other
        self.bn_rssi = nn.BatchNorm1d(embedding_dim)
        self.bn_features = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(0.3),

    def forward(self, x):
        embd_bssid = self.embd_bssid(x['BSSID_FEATS'])  # (,wifi_num,64)

        embd_site_id = self.embd_site_id(x['site_id'])  # (,2)
        embd_site_id = torch.unsqueeze(embd_site_id, dim=1)  # (,1,2)
        embd_site_id = embd_site_id.repeat(
            1, self.wifi_num, 1)  # (,wifi_num,2)

        rssi_feat = x['RSSI_FEATS']  # (,wifi_num)
        rssi_feat = torch.unsqueeze(rssi_feat, dim=-1)   # (,wifi_num,1)
        rssi_feat = self.fc_rssi(rssi_feat)              # (,wifi_num,64)
        rssi_feat = self.bn_rssi(rssi_feat.transpose(1, 2)).transpose(1, 2)
        rssi_feat = torch.relu(rssi_feat)

        x = torch.cat([embd_bssid, embd_site_id, rssi_feat],
                      dim=-1)  # (,wifi_num,feature_dim)

        regression_token = self.reg_token.repeat(
            x.shape[0], 1, 1)  # (,1,feature_dim)

        # (,wifi_num+1,feature_dim)
        x = torch.cat([regression_token, x], dim=1)
        x = self.fc_features(x)  # (,wifi_num+1, 128)
        x = self.bn_features(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)

        x = torch.transpose(x, 0, 1)  # (wifi_num+1,, 128)
        x = self.transformer_encoder(x, )  # (wifi_num+1,,128)

        x = x[0]  # Regression output  # (1,,128)

        output = self.fc_output(x).squeeze()  # (,3)

        return output
