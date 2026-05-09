import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.losses import QuantileLoss


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads,
        num_layers,
        encoder_length,
        decoder_length,
        quantiles,
        lr=1e-3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, len(quantiles))

        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.loss_fn = QuantileLoss(quantiles)
        self.lr = lr

    def forward(self, x_enc, x_dec):
        x = torch.cat([x_enc, x_dec], dim=1)
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        out = out[:, -self.decoder_length :, :]
        return self.fc_out(out)

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