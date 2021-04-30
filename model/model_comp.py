import torch.nn as nn
import torch
import numpy as np
import neptune
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from torch import optim
from icecream import ic
from config import Config


def metric_comp(x_hat, y_hat, f_hat, x, y, f):
    intermediate = np.sqrt((x_hat-x)**2 + (y_hat-y)**2) + 15 * np.abs(f_hat-f)
    return intermediate.sum()/x_hat.shape[0]


def xy_loss(xy_hat, xy_label):
    xy_loss = torch.mean(torch.sqrt(
        (xy_hat[:, 0]-xy_label[:, 0])**2 + (xy_hat[:, 1]-xy_label[:, 1])**2))
    return xy_loss


def floor_loss(floor_hat, floor_label):
    floor_loss = 15 * torch.mean(torch.abs(floor_hat-floor_label))
    return floor_loss


class IndoorLocModel(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.lr = Config().lr

        self.critertion_xy = xy_loss
        self.criterion_floor = floor_loss

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']

        xy_label = torch.cat([x, y], dim=-1)

        output = self(batch)
        xy_hat = output[:, 0:2]
        f_hat = output[:, 2]

        loss_xy = self.critertion_xy(xy_hat, xy_label)
        loss_floor = self.criterion_floor(f_hat, f)
        loss = loss_xy + loss_floor

        return {'loss': loss, 'loss_xy': loss_xy, 'loss_floor': loss_floor, 'xy_label': xy_label, 'xy_hat': xy_hat, 'floor_hat': f_hat, 'f': f}

    def training_epoch_end(self, outputs):
        loss_xy = torch.mean(torch.stack(
            [output['loss_xy'] for output in outputs], dim=0))
        loss_floor = torch.mean(torch.stack(
            [output['loss_floor'] for output in outputs], dim=0))
        loss = torch.mean(torch.stack([output['loss']
                          for output in outputs], dim=0))

        if Config.neptune:
            neptune.log_metric('train_loss', loss)
            neptune.log_metric('train_loss_xy', loss_xy)
            neptune.log_metric('train_loss_floor', loss_floor)

    def validation_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']

        xy_label = torch.cat([x, y], dim=-1)

        output = self(batch)
        xy_hat = output[:, 0:2]
        f_hat = output[:, 2]

        return {'xy_label': xy_label, 'xy_hat': xy_hat, 'f_hat': f_hat, 'f': f}

    def validation_epoch_end(self, outputs):
        xy_label = torch.cat([output['xy_label'] for output in outputs], dim=0)
        xy_hat = torch.cat([output['xy_hat'] for output in outputs], dim=0)
        f_hat = torch.cat([output['f_hat']
                           for output in outputs], dim=0)
        f_hat = torch.squeeze(f_hat)
        f = torch.cat([output['f'] for output in outputs], dim=0)

        loss_xy = self.critertion_xy(xy_hat, xy_label)
        loss_floor = self.criterion_floor(f_hat, f)
        loss = loss_xy + loss_floor

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', loss, prog_bar=True)

        if Config.neptune:
            neptune.log_metric('val_loss', loss)
            neptune.log_metric('val_loss_xy', loss_xy)
            neptune.log_metric('val_loss_floor', loss_floor)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # optimizer = optim.RAdam(
        #     model.parameters(),
        #     lr= 1e-3,
        #     betas=(0.9, 0.999),
        #     eps=1e-8,
        #     weight_decay=0,
        # )
        # optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=10, eta_min=0)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss', }
