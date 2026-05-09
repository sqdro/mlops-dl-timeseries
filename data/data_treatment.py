"""Módulo para cargar, procesar y preparar datos de series temporales"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch

from src.utils import load_config


class TimeSeriesDataset(Dataset):
    """Dataset para series temporales con estructura encoder-decoder"""
    
    def __init__(self, df, features, target, encoder_length=30, decoder_length=7):
        self.X = df[features].values
        self.y = df[target].values
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

    def __len__(self):
        return len(self.X) - self.encoder_length - self.decoder_length + 1

    def __getitem__(self, idx):
        x_enc = self.X[idx : idx + self.encoder_length]
        x_dec = self.X[idx + self.encoder_length : idx + self.encoder_length + self.decoder_length]
        y_dec = self.y[idx + self.encoder_length : idx + self.encoder_length + self.decoder_length]

        return {
            "encoder_x": torch.tensor(x_enc, dtype=torch.float),
            "decoder_x": torch.tensor(x_dec, dtype=torch.float),
            "decoder_y": torch.tensor(y_dec, dtype=torch.float)
        }


def load_dataset(data_path):
    """Cargar Dataset desde CSV"""

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.drop(columns=["id"]).sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] / 12))
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] / 12))

    return df


def split_and_scale_data(df, features, target, train_ratio):
    """Dividir el dataset en train/val y escalar usando MinMaxScaler"""

    train_size = int(len(df) * train_ratio)

    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    df_train[features] = scaler_X.fit_transform(df_train[features])
    df_val[features] = scaler_X.transform(df_val[features])

    df_train[target] = scaler_y.fit_transform(df_train[[target]])
    df_val[target] = scaler_y.transform(df_val[[target]])

    return df_train, df_val, scaler_X, scaler_y


def build_dataloaders(df_train, df_val, features, target, encoder_length, decoder_length, batch_size):
    """Construir datasets y dataloaders para train y validación"""

    train_dataset = TimeSeriesDataset(
        df_train,
        features,
        target,
        encoder_length=encoder_length,
        decoder_length=decoder_length
    )
    val_dataset = TimeSeriesDataset(
        df_val,
        features,
        target,
        encoder_length=encoder_length,
        decoder_length=decoder_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader


def prepare_data(config_path="config.yaml"):
    """Preparar datos para entrenamiento: cargar, dividir, escalar y construir dataloaders"""

    args = load_config(config_path)

    # Cargar dataset
    df = load_dataset(args["data_path"])

    # Split y escalado
    df_train, df_val, scaler_X, scaler_y = split_and_scale_data(
        df,
        args["features"],
        args["target"],
        args["train_ratio"]
    )

    # Construir DataLoaders
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        df_train, 
        df_val, 
        args["features"],
        args["target"],
        args["encoder_length"],
        args["decoder_length"],
        args["batch_size"]
    )

    return {
        "df": df,
        "df_train": df_train,
        "df_val": df_val,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader
    }
