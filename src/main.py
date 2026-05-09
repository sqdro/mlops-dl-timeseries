"""Módulo principal para ejecutar el pipeline de entrenamiento de modelos de series temporales de Deep Learning"""

import wandb
import argparse
import numpy as np
from pathlib import Path

from models.mlp import MLPModel
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.tcn import TCNModel
from models.transformer import TransformerModel

from data.data_treatment import prepare_data
from src.predict import desnormalize_preds, predict_quantiles
from src.train import train_model
from src.utils import load_config, setup_logger
from src.plots import plot_prediction


def build_mlp(input_size, encoder_length, decoder_length, quantiles, cfg):
    """Construir modelo MLP"""
    return MLPModel(
        input_size=input_size,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        quantiles=quantiles,
        **cfg
    )


def build_lstm(input_size, encoder_length, decoder_length, quantiles, cfg):
    """Construir modelo LSTM"""
    return LSTMModel(
        input_size=input_size,
        decoder_length=decoder_length,
        quantiles=quantiles,
        **cfg
    )


def build_gru(input_size, encoder_length, decoder_length, quantiles, cfg):
    """Construir modelo GRU"""
    return GRUModel(
        input_size=input_size,
        decoder_length=decoder_length,
        quantiles=quantiles,
        **cfg
    )


def build_tcn(input_size, encoder_length, decoder_length, quantiles, cfg):
    """Construir modelo TCN"""
    return TCNModel(
        input_size=input_size,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        quantiles=quantiles,
        **cfg
    )


def build_transformer(input_size, encoder_length, decoder_length, quantiles, cfg):
    """Construir modelo Transformer"""
    return TransformerModel(
        input_size=input_size,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        quantiles=quantiles,
        **cfg
    )


MODEL_BUILDERS = {
    "mlp": build_mlp,
    "lstm": build_lstm,
    "gru": build_gru,
    "tcn": build_tcn,
    "transformer": build_transformer
}


def build_model(model_to_train, encoder_length, decoder_length, quantiles, model_configs, features):
    """Construir modelo especificado en la configuración"""

    model_key = model_to_train.lower()

    if model_key not in MODEL_BUILDERS:
        valid_models = ", ".join(sorted(MODEL_BUILDERS.keys()))
        logger = setup_logger(__name__)
        logger.error(f"Modelo '{model_to_train}' no es válido. Modelos disponibles: {valid_models}")
        raise ValueError(f"Modelo '{model_to_train}' no es válido. Modelos disponibles: {valid_models}")

    if model_key not in model_configs:
        logger = setup_logger(__name__)
        logger.error(f"No existe configuración para '{model_key}'")
        raise KeyError(f"No existe configuración para '{model_key}'")

    input_size = len(features)
    model = MODEL_BUILDERS[model_key](
        input_size=input_size,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        quantiles=quantiles,
        cfg=model_configs[model_key]
    )

    return model_key.upper(), model


def main():
    """Pipeline principal de entrenamiento"""

    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento para modelos de series temporales de Deep Learning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración YAML")
    args = parser.parse_args()

    # Configurar log
    logger = setup_logger(__name__)
    logger.info("Iniciando pipeline de entrenamiento")

    # Cargar configuración
    logger.info("Cargando configuración")
    config = load_config(args.config)

    # Inicializar Weights & Biases
    logger.info("Inicializando Weights & Biases")
    wandb.init(
        project="TimeSeriesForecasting",
        name=config["experiment_name"],
        config=config,
        job_type="training"
    )

    # Preparar datos
    logger.info("Cargando configuración y preparando datos")
    data_artifacts = prepare_data(config_path=args.config)
    trained_models = {}

    # Construir modelo
    logger.info(f"Construyendo modelo: {config['model_to_train']}")
    model_name, model = build_model(
        model_to_train=config["model_to_train"],
        encoder_length=config["encoder_length"],
        decoder_length=config["decoder_length"],
        quantiles=config["quantiles"],
        model_configs=config["model_configs"],
        features=config["features"],
    )

    # Entrenar modelo
    logger.info(f"Entrenando modelo: {model_name}")
    trained_model, trainer = train_model(
        model,
        train_loader=data_artifacts["train_loader"],
        val_loader=data_artifacts["val_loader"],
        dir_checkpoint=config["checkpoint_dir"],
        model_name=model_name.lower(),
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        min_delta=config["min_delta"],
        gradient_clip_val=config["gradient_clip_val"],
        save_top_k=config.get("save_top_k", 1)
    )
    trained_models[model_name] = trained_model
    logger.info(f"Modelo {model_name} entrenado correctamente")

    # Guardar el modelo entrenado como un artefacto de Weights & Biases
    logger.info(f"Guardando modelo {model_name} como artefacto en Weights & Biases")
    artifact_model = wandb.Artifact(
        name=f"{model_name}_model", 
        type="model", 
        description=f"Modelo {model_name} entrenado para series temporales"
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    artifact_model.add_file(best_model_path)
    wandb.log_artifact(artifact_model)

    # Generar predicciones y desnormalizarlas
    logger.info(f"Generando predicciones para: {model_name}")
    preds_scaled = predict_quantiles(trained_model, dataset=data_artifacts["val_dataset"])
    y_pred = desnormalize_preds(preds_scaled, data_artifacts["scaler_y"])

    # Crear plots y guardarlos como artefactos de Weights & Biases
    logger.info("Generando gráficos de predicción")
    plot_dir = Path(config.get("plot_dir", "logs/plots"))
    plot_dir.mkdir(exist_ok=True)
    
    # Obtener toda la serie temporal del dataset de validación
    y_true_list = []
    for i in range(len(data_artifacts["val_dataset"])):
        y_true_sample = data_artifacts["val_dataset"][i]["decoder_y"].numpy().reshape(-1, 1)
        y_true_desnorm = data_artifacts["scaler_y"].inverse_transform(y_true_sample).flatten()
        y_true_list.append(y_true_desnorm)
    y_true = np.array(y_true_list)
    
    plot_path = plot_dir / f"{model_name.lower()}_prediction.png"

    plot_prediction(y_true=y_true, y_pred=y_pred, quantiles=config["quantiles"], 
                    title=f"Predicción - {model_name}", save_path=plot_path
    )

    # Guardar el plot como artefacto en Weights & Biases
    logger.info(f"Guardando gráfico de predicción para {model_name} como artefacto en Weights & Biases")
    artifact_plots = wandb.Artifact(
        name=f"{model_name}_plots",
        type="plots",
        description=f"Plots de predicción para el modelo {model_name}"
    )
    artifact_plots.add_file(str(plot_path))
    wandb.log_artifact(artifact_plots)

    logger.info("Pipeline completado")
    wandb.finish()
    
    return trained_models, data_artifacts


if __name__ == "__main__":
    main()
