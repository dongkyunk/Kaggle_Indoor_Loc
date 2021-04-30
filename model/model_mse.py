import torch.nn as nn
import torch
import numpy as np
import neptune
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from torch import optim
from icecream import ic
from config import Config


def xy_metric(xy_hat, xy_label):
    xy_loss = torch.mean(torch.sqrt(
        (xy_hat[:, 0]-xy_label[:, 0])**2 + (xy_hat[:, 1]-xy_label[:, 1])**2))
    return xy_loss


def floor_metric(floor_hat, floor_label):
    floor_loss = 15 * torch.mean(torch.abs(floor_hat-floor_label))
    return floor_loss


class IndoorLocModel(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.lr = Config.lr

        self.critertion_xy = nn.MSELoss()
        self.criterion_floor = nn.MSELoss()

        self.metric_xy = xy_metric
        self.metric_floor = floor_metric

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, f = batch['x'], batch['y'], batch['floor']

        xy_label = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

        output = self(batch)

        xy_hat = output[:, 0:2]
        f_hat = output[:, 2]

        loss_xy = self.critertion_xy(xy_hat.float(), xy_label.float())
        loss_floor = 2250 * self.criterion_floor(f_hat.float(), f.float())
        loss = loss_xy + loss_floor

        metric_xy = self.metric_xy(xy_hat, xy_label)
        metric_floor = self.metric_floor(f_hat, f)
        metric = metric_xy + metric_floor

       return {'loss': loss, 'loss_xy': loss_xy, 'loss_floor': loss_floor, 'metric_xy': metric_xy, 'metric_floor': metric_floor, 'metric': metric}

    def training_epoch_end(self, outputs):
        loss_xy = torch.mean(torch.stack(
            [output['loss_xy'] for output in outputs], dim=0))
        loss_floor = torch.mean(torch.stack(
            [output['loss_floor'] for output in outputs], dim=0))
        loss = torch.mean(torch.stack([output['loss']
                          for output in outputs], dim=0))
        metric_xy = torch.mean(torch.stack([output['metric_xy']
                                            for output in outputs], dim=0))
        metric_floor = torch.mean(torch.stack([output['metric_floor']
                                               for output in outputs], dim=0))
        metric = torch.mean(torch.stack([output['metric']
                                         for output in outputs], dim=0))

        if Config.neptune:
            neptune.log_metric('train_loss', loss)
            neptune.log_metric('train_loss_xy', loss_xy)
            neptune.log_metric('train_loss_floor', loss_floor)
            neptune.log_metric('train_metric', metric)
            neptune.log_metric('train_metric_floor', metric_floor)
            neptune.log_metric('train_metric_xy', metric_xy)

    def validation_step(self, batch, batch_nb):
        x, y, f = batch['x'], batch['y'], batch['floor']

        xy_label = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

        output = self(batch)

        xy_hat = output[:, 0:2]
        f_hat = output[:, 2]

        loss_xy = self.critertion_xy(xy_hat.float(), xy_label.float())
        loss_floor = 2250 * self.criterion_floor(f_hat.float(), f.float())
        loss = loss_xy + loss_floor

        metric_xy = self.metric_xy(xy_hat, xy_label)
        metric_floor = self.metric_floor(f_hat, f)
        metric = metric_xy + metric_floor

        return {'loss': loss, 'loss_xy': loss_xy, 'loss_floor': loss_floor, 'metric_xy': metric_xy, 'metric_floor': metric_floor, 'metric': metric}

    def validation_epoch_end(self, outputs):
        loss_xy = torch.mean(torch.stack(
            [output['loss_xy'] for output in outputs], dim=0))
        loss_floor = torch.mean(torch.stack(
            [output['loss_floor'] for output in outputs], dim=0))
        loss = torch.mean(torch.stack([output['loss']
                          for output in outputs], dim=0))
        metric_xy = torch.mean(torch.stack([output['metric_xy']
                                            for output in outputs], dim=0))
        metric_floor = torch.mean(torch.stack([output['metric_floor']
                                               for output in outputs], dim=0))
        metric = torch.mean(torch.stack([output['metric']
                                         for output in outputs], dim=0))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=True)
        self.log('val_loss_xy', loss_xy, prog_bar=True)
        self.log('val_loss_floor', loss_floor, prog_bar=True)
        self.log('val_metric_floor', metric_floor, prog_bar=True)
        self.log('val_metric_xy', metric_xy, prog_bar=True)

        if Config.neptune:
            neptune.log_metric('val_loss', loss)
            neptune.log_metric('val_loss_xy', loss_xy)
            neptune.log_metric('val_loss_floor', loss_floor)
            neptune.log_metric('val_metric', metric)
            neptune.log_metric('val_metric_floor', metric_floor)
            neptune.log_metric('val_metric_xy', metric_xy)

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
