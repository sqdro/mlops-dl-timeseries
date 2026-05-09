import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.losses import QuantileLoss


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, decoder_length, quantiles, lr=1e-3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(quantiles))

        self.loss_fn = QuantileLoss(quantiles)
        self.lr = lr
        self.decoder_length = decoder_length

    def forward(self, x_enc, x_dec):
        _, (hidden_state, cell_state) = self.encoder(x_enc)
        out_dec, _ = self.decoder(x_dec, (hidden_state, cell_state))
        return self.fc(out_dec)

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