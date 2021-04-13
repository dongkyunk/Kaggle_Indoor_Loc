import torch
import torch.nn as nn


class simpleLSTM(nn.Module):
    def __init__(self, input_dim, bssid_dim, site_id_dim, embedding_dim=64, seq_len=20):
        super(simpleLSTM, self).__init__()
        # Embedding
        self.embd_bssid = nn.Embedding(bssid_dim, embedding_dim)
        self.embd_site_id = nn.Embedding(site_id_dim, 2)

        # LSTM
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128,
                             dropout=0.3, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=16,
                             dropout=0.1, bidirectional=False)
        
        self.fc_rssi = nn.Linear(input_dim, input_dim * embedding_dim)
        self.fc_features = nn.Linear(2562, 256)
        self.lr_xy = nn.Linear(16, 2)
        self.lr_floor = nn.Linear(16, 1)
        
        self.batch_norm_rssi = nn.BatchNorm1d(input_dim)
        self.batch_norm2 = nn.BatchNorm1d(2562)
        self.batch_norm3 = nn.BatchNorm1d(1)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embd_bssid = self.embd_bssid(x['BSSID_FEATS']) # (,,input_dim) -> (,,64)
        embd_bssid = torch.flatten(embd_bssid, start_dim=1)

        embd_site_id = self.embd_site_id(x['site_id']) # (,,2)
        embd_site_id = torch.flatten(embd_site_id, start_dim=1) # (,,2)

        rssi_feat = self.batch_norm1(x['RSSI_FEATS']) #(,,)
        rssi_feat = self.rssi_fc(rssi_feat)
        rssi_feat = torch.relu(rssi_feat)

        x = torch.cat([bssid_embd, emb_site_id, rssi_feat], dim=-1)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = torch.relu(self.lr1(x))

        x = x.unsqueeze(-2)
        x = self.batch_norm3(x)
        x = x.transpose(0, 1)
        x, _ = self.lstm1(x)
        x = x.transpose(0, 1)
        x = torch.relu(x)
        x = x.transpose(0, 1)
        x, _ = self.lstm2(x)
        x = x.transpose(0, 1)
        x = torch.relu(x)
        xy = self.lr_xy(x)
        floor = self.lr_floor(x)
        floor = torch.relu(floor)

        return xy.squeeze(-2), floor.squeeze(-2)

sl = simpleLSTM(20, 61206, 1000)
print(sl)