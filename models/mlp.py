import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.losses import QuantileLoss


class MLPModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, encoder_length, decoder_length, quantiles, lr=1e-3):
        super().__init__()

        self.hidden_size = hidden_size

        total_input = input_size * (encoder_length + decoder_length)

        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, decoder_length * len(quantiles)),
        )

        self.decoder_length = decoder_length
        self.quantiles = quantiles
        self.loss_fn = QuantileLoss(quantiles)
        self.lr = lr

    def forward(self, x_enc, x_dec):
        x = torch.cat([x_enc, x_dec], dim=1)
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = out.view(-1, self.decoder_length, len(self.quantiles))
        return out

    def training_step(self, batch, batch_idx):
        y_hat = self(batch["encoder_x"], batch["decoder_x"])
        loss = self.loss_fn(y_hat, batch["decoder_y"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch["encoder_x"], batch["decoder_x"])
        loss = self.loss_fn(y_hat, batch["decoder_y"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)