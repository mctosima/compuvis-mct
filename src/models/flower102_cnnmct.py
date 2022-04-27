from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.convmct import ConvMCT


class Flowers102CNNMCT(LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        loss_fn=torch.nn.CrossEntropyLoss,
        weight_decay: float = 0.0,
    ):

        super().__init__()
        self.net = ConvMCT()
        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimz = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimz

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        print(logits.shape)
        print(y.shape)
        print(self.hparams.loss_fn)
        loss = self.hparams.loss_fn(logits, y - 1)
        y_hat = torch.argmax(logits, dim=1)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch)
        acc = self.train_acc(y_hat, y)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch)
        acc = self.val_acc(y_hat, y)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log(
            "val_acc_best", self.val_acc_best.compute(), on_step=True, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch)
        acc = self.test_acc(y_hat, y)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_epoch_end(self, outputs):
        pass

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
