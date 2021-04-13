import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule

class Encoder(nn.Module):
    '''
    CNN Feature Vector Extraction
    '''
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.cnn.fc.in_features
        self.cnn.global_pool = nn.Identity()
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        bs = x.size(0)
        features = self.cnn(x)
        features = features.permute(0, 2, 3, 1)
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        '''
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        '''
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    def

# class MnistModel(LightningModule):
#     def __init__(self, model: nn.Module = None):
#         super().__init__()
#         self.model = model
#         self.criterion = nn.MultiLabelSoftMarginLoss()
#         self.metric = Accuracy(threshold=0.5)

#     def forward(self, x):
#         x = self.model(x)
#         return x

#     def training_step(self, batch, batch_nb):
#         x, y = batch
#         y_hat = self(x)

#         loss = self.criterion(y_hat, y)
#         acc = self.metric(y_hat, y)

#         neptune.log_metric('train_loss', loss)
#         neptune.log_metric('train_accuracy', acc)

#         return loss     

#     def validation_step(self, batch, batch_nb):
#         x, y = batch
#         y_hat = self(x)

#         loss = self.criterion(y_hat, y)
#         acc = self.metric(y_hat, y)

#         return {'loss': loss, 'acc': acc}


#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

#         self.log('val_loss', avg_loss, prog_bar=True)
#         self.log('val_acc', avg_acc, prog_bar=True)

#         neptune.log_metric('val_loss', avg_loss)
#         neptune.log_metric('val_acc', avg_acc)   


#     def configure_optimizers(self):
#         # optimizer = optim.Adam(model.parameters(), lr=1e-3)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#         optimizer = optim.RAdam(
#             model.parameters(),
#             lr= 1e-3,
#             betas=(0.9, 0.999),
#             eps=1e-8,
#             weight_decay=0,
#         )
#         optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=10, eta_min=0)

#         return [optimizer], [scheduler]