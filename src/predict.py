"""Módulo para realizar predicciones con modelos entrenados"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def predict_quantiles(model, dataset, batch_size=64):
    """Generar predicciones de cuantiles para un dataset dado usando el modelo entrenado"""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_enc = batch["encoder_x"].to(model.device)
            x_dec = batch["decoder_x"].to(model.device)
            y_hat = model(x_enc, x_dec).cpu().numpy()
            all_preds.append(y_hat)

    return np.concatenate(all_preds, axis=0)


def desnormalize_preds(y_scaled, scaler_y):
    """Desnormalizar predicciones para obtener valores en la escala original"""

    shape = y_scaled.shape
    y_flat = y_scaled.reshape(-1, y_scaled.shape[-1])
    y_real_flat = scaler_y.inverse_transform(y_flat)
    
    return y_real_flat.reshape(shape)
