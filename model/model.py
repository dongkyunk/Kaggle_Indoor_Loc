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


class IndoorLocModel(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.lr = Config().lr

        self.critertion_xy = nn.MSELoss()
        self.criterion_floor = nn.CrossEntropyLoss()
        self.metric_comp = metric_comp
        self.metric_floor = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']
        f += 2  # Floor starts from -2

        xy_label = torch.cat([x, y], dim=-1)

        xy_hat, floor_hat = self(batch)

        loss_xy = self.critertion_xy(xy_hat.float(), xy_label.float())
        # ic(xy_hat)
        # ic(xy_hat.float())
        # ic(xy_label.float())
        loss_floor = self.criterion_floor(floor_hat, f)
        loss = loss_xy + 5000 * loss_floor

        metric = self.metric_comp(xy_hat[:, 0].cpu().detach(), xy_hat[:, 1].cpu().detach(
        ), torch.argmax(floor_hat, dim=-1).cpu().detach(), x.cpu(), y.cpu(), f.cpu())
        metric_floor = self.metric_floor(floor_hat, f)

        if Config.neptune:
            neptune.log_metric('train_loss', loss)
            neptune.log_metric('train_loss_xy', loss_xy)
            neptune.log_metric('train_loss_floor', loss_floor)
            neptune.log_metric('train_metric', metric)
            neptune.log_metric('train_floor_accuracy', metric_floor)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']
        f += 2  # Floor starts from -2

        xy_label = torch.cat([x, y], dim=-1)

        xy_hat, floor_hat = self(batch)

        loss_xy = self.critertion_xy(xy_hat.float(), xy_label.float())
        loss_floor = self.criterion_floor(floor_hat, f)
        loss = loss_xy + 5000 * loss_floor

        metric = self.metric_comp(xy_hat[:, 0].cpu().detach(), xy_hat[:, 1].cpu().detach(
        ), torch.argmax(floor_hat, dim=-1).cpu().detach(), x.cpu(), y.cpu(), f.cpu())
        metric_floor = self.metric_floor(floor_hat, f)

        return {'loss': loss, 'loss_xy': loss_xy, 'loss_floor': loss_floor, 'metric': metric, 'metric_floor': metric_floor}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_xy = torch.stack([x['loss_xy'] for x in outputs]).mean()
        avg_loss_floor = torch.stack([x['loss_floor'] for x in outputs]).mean()
        avg_metric = torch.stack([x['metric'] for x in outputs]).mean()
        avg_metric_floor = torch.stack(
            [x['metric_floor'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_metric', avg_metric, prog_bar=True)

        if Config.neptune:
            neptune.log_metric('val_loss', avg_loss)
            neptune.log_metric('val_loss_xy', avg_loss_xy)
            neptune.log_metric('val_loss_floor', avg_loss_floor)
            neptune.log_metric('val_metric', avg_metric)
            neptune.log_metric('val_floor_accuracy', avg_metric_floor)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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

        return optimizer
