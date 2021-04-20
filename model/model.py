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

def xy_loss(xy_hat, xy_label)
    xy_loss = torch.mean(torch.sqrt((xy_hat[:, 0]-xy_label[:, 0])**2 + (xy_hat[:, 1]-xy_label[:, 1])**2))
    return xy_loss

def floor_loss(floor_hat, floor_label):
    l1 = nn.L1Loss()
    floor_loss = 15 * l1(floor_hat, floor_label) 
    return floor_loss

class IndoorLocModel(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.lr = Config().lr

        # self.critertion_xy = nn.MSELoss()
        # self.criterion_floor = nn.CrossEntropyLoss()
        self.critertion_xy = xy_loss
        self.criterion_floor = floor_loss
        self.metric_comp = torch_metric_comp
        self.metric_floor = Accuracy()

        self.loss_floor_weight = 0
        self.loss_xy_weight = 1
        self.training_phase = "xy"

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
        loss_floor = self.criterion_floor(floor_hat, f)
        loss = self.loss_xy_weight * loss_xy + self.loss_floor_weight * loss_floor

        return {'loss': loss, 'xy_label': xy_label, 'xy_hat': xy_hat, 'floor_hat': floor_hat, 'f': f}

    def training_epoch_end(self, outputs):
        xy_label = torch.cat([output['xy_label'] for output in outputs], dim=0)
        xy_hat = torch.cat([output['xy_hat'] for output in outputs], dim=0)
        floor_hat = torch.cat([output['floor_hat']
                              for output in outputs], dim=0)
        f = torch.cat([output['f'] for output in outputs], dim=0)

        loss_xy = self.critertion_xy(xy_hat.float(), xy_label.float())
        loss_floor = self.criterion_floor(floor_hat, f)

        loss = self.loss_xy_weight * loss_xy + self.loss_floor_weight * loss_floor

        metric = self.metric_comp(xy_hat[:, 0].cpu().detach(), xy_hat[:, 1].cpu().detach(
        ), torch.argmax(floor_hat, dim=-1).cpu().detach(), xy_label[:, 0].cpu(), xy_label[:, 1].cpu(), f.cpu())
        metric_floor = self.metric_floor(floor_hat, f)

        if Config.neptune:
            neptune.log_metric('train_loss', loss)
            neptune.log_metric('train_loss_xy', loss_xy)
            neptune.log_metric('train_loss_floor', loss_floor)
            neptune.log_metric('train_metric', metric)
            neptune.log_metric('train_floor_accuracy', metric_floor)
            neptune.log_metric('loss_floor_weight', self.loss_floor_weight)

    def validation_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']
        f += 2  # Floor starts from -2

        xy_label = torch.cat([x, y], dim=-1)

        xy_hat, floor_hat = self(batch)

        return {'xy_label': xy_label, 'xy_hat': xy_hat, 'floor_hat': floor_hat, 'f': f}

    def validation_epoch_end(self, outputs):
        xy_label = torch.cat([output['xy_label'] for output in outputs], dim=0)
        xy_hat = torch.cat([output['xy_hat'] for output in outputs], dim=0)
        floor_hat = torch.cat([output['floor_hat']
                              for output in outputs], dim=0)
        f = torch.cat([output['f'] for output in outputs], dim=0)

        loss_xy = self.critertion_xy(xy_hat.float(), xy_label.float())
        loss_floor = self.criterion_floor(floor_hat, f)
        loss = self.loss_xy_weight*loss_xy + self.loss_floor_weight * loss_floor

        metric = self.metric_comp(xy_hat[:, 0].cpu().detach(), xy_hat[:, 1].cpu().detach(
        ), torch.argmax(floor_hat, dim=-1).cpu().detach(), xy_label[:, 0].cpu(), xy_label[:, 1].cpu(), f.cpu())
        metric_floor = self.metric_floor(floor_hat, f)

        # Train xy only when val floor acc is above 99%
        if (loss_xy < 20) and (metric_floor < 0.99):
            self.training_phase = "floor"
        if metric_floor > 0.99:
            self.training_phase = "xy"

        if self.training_phase == "floor":
            self.loss_floor_weight = 1
            self.loss_xy_weight = 0
        elif self.training_phase == "xy":
            self.loss_floor_weight = 0
            self.loss_xy_weight = 1

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=True)

        if Config.neptune:
            neptune.log_metric('val_loss', loss)
            neptune.log_metric('val_loss_xy', loss_xy)
            neptune.log_metric('val_loss_floor', loss_floor)
            neptune.log_metric('val_metric', metric)
            neptune.log_metric('val_floor_accuracy', metric_floor)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
