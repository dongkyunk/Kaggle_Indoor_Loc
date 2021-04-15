import torch.nn as nn
import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch import optim
from icecream import ic


def comp_metric(x_hat, y_hat, f_hat, x, y, f):
    intermediate = np.sqrt((x_hat-x)**2 + (y_hat-y)**2) + 15 * np.abs(f_hat-f)
    return intermediate.sum()/x_hat.shape[0]


class IndoorLocModel(LightningModule):
    def __init__(self, model: nn.Module = None):
        super().__init__()
        self.model = model
        self.critertion_xy = nn.MSELoss()
        self.criterion_floor = nn.CrossEntropyLoss()
        self.metric = comp_metric

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']
        f += 2  # Floor starts from -2

        xy_label = torch.cat([x, y], dim=-1)

        xy_hat, floor_hat = self(batch)

        loss_xy = self.critertion_xy(xy_hat, xy_label)
        loss_floor = self.criterion_floor(floor_hat, f).float()
        loss = (loss_xy + 5000 * loss_floor).type(torch.float32)

        ic(loss_xy)
        ic(loss_floor)
        ic(loss)

        metric = self.metric(xy_hat[:, 0].detach(), xy_hat[:, 1].detach(
        ), torch.argmax(floor_hat, dim=-1).detach(), x, y, f)

        #neptune.log_metric('train_loss', loss)
        #neptune.log_metric('train_metric', metric)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']
        f += 2  # Floor starts from -2

        xy_label = torch.cat([x, y], dim=-1)

        xy_hat, floor_hat = self(batch)

        loss_xy = self.critertion_xy(xy_hat, xy_label)
        loss_floor = self.criterion_floor(floor_hat, f).float()
        loss = (loss_xy + 5000 * loss_floor).type(torch.float32)

        ic(loss_xy)
        ic(loss_floor)
        ic(loss)

        metric = self.metric(xy_hat[:, 0].detach(), xy_hat[:, 1].detach(
        ), torch.argmax(floor_hat, dim=-1).detach(), x, y, f)

        return {'loss': loss, 'metric': metric}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_metric = torch.stack([x['metric'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_metric', avg_metric, prog_bar=True)

        #neptune.log_metric('val_loss', avg_loss)
        #neptune.log_metric('val_metric', avg_metric)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-3)

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

        return [optimizer]
