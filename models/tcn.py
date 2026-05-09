import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.losses import QuantileLoss


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, : -(self.conv.padding[0])]
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TCNModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, encoder_length, decoder_length, quantiles, lr=1e-3):
        super().__init__()

        self.hidden_size = hidden_size

        self.tcn1 = TCNBlock(input_size, hidden_size, kernel_size=3, dilation=1)
        self.tcn2 = TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=2)
        self.fc = nn.Linear(hidden_size, len(quantiles))

        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.loss_fn = QuantileLoss(quantiles)
        self.lr = lr

    def forward(self, x_enc, x_dec):
        x = torch.cat([x_enc, x_dec], dim=1)
        x = x.transpose(1, 2)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x[:, :, -self.decoder_length :]
        x = x.transpose(1, 2)
        return self.fc(x)

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