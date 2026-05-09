"""API de Inferencia para Modelos de Series Temporales con FastAPI"""

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel, Field
import torch
import numpy as np
from pathlib import Path
from typing import List

from src.utils import load_config, get_project_root, setup_logger
from models.lstm import LSTMModel


logger = setup_logger(__name__, log_file="api.log")


class EncoderInput(BaseModel):
    """Datos de entrada para el encoder (serie temporal histórica)"""
    features: List[List[float]] = Field(
        description="Matriz de features históricos (shape: encoder_length x num_features)",
        examples=[
            [
                [22.5, 18.3, 19.2, 20.1, 21.0, 19.5, 0.5, 0.866],
                [22.6, 18.4, 19.3, 20.2, 21.1, 19.6, 0.5, 0.866],
                [22.7, 18.5, 19.4, 20.3, 21.2, 19.7, 0.5, 0.866],
                [22.8, 18.6, 19.5, 20.4, 21.3, 19.8, 0.5, 0.866],
                [22.9, 18.7, 19.6, 20.5, 21.4, 19.9, 0.5, 0.866],
                [23.0, 18.8, 19.7, 20.6, 21.5, 20.0, 0.5, 0.866],
                [23.1, 18.9, 19.8, 20.7, 21.6, 20.1, 0.5, 0.866],
                [23.2, 19.0, 19.9, 20.8, 21.7, 20.2, 0.5, 0.866],
                [23.3, 19.1, 20.0, 20.9, 21.8, 20.3, 0.5, 0.866],
                [23.4, 19.2, 20.1, 21.0, 21.9, 20.4, 0.5, 0.866],
                [23.5, 19.3, 20.2, 21.1, 22.0, 20.5, 0.5, 0.866],
                [23.6, 19.4, 20.3, 21.2, 22.1, 20.6, 0.5, 0.866],
                [23.7, 19.5, 20.4, 21.3, 22.2, 20.7, 0.5, 0.866],
                [23.8, 19.6, 20.5, 21.4, 22.3, 20.8, 0.5, 0.866],
                [23.9, 19.7, 20.6, 21.5, 22.4, 20.9, 0.5, 0.866],
                [24.0, 19.8, 20.7, 21.6, 22.5, 21.0, 0.5, 0.866],
                [24.1, 19.9, 20.8, 21.7, 22.6, 21.1, 0.5, 0.866],
                [24.2, 20.0, 20.9, 21.8, 22.7, 21.2, 0.5, 0.866],
                [24.3, 20.1, 21.0, 21.9, 22.8, 21.3, 0.5, 0.866],
                [24.4, 20.2, 21.1, 22.0, 22.9, 21.4, 0.5, 0.866],
                [24.5, 20.3, 21.2, 22.1, 23.0, 21.5, 0.5, 0.866],
                [24.6, 20.4, 21.3, 22.2, 23.1, 21.6, 0.5, 0.866],
                [24.7, 20.5, 21.4, 22.3, 23.2, 21.7, 0.5, 0.866],
                [24.8, 20.6, 21.5, 22.4, 23.3, 21.8, 0.5, 0.866],
                [24.9, 20.7, 21.6, 22.5, 23.4, 21.9, 0.5, 0.866],
                [25.0, 20.8, 21.7, 22.6, 23.5, 22.0, 0.5, 0.866],
                [25.1, 20.9, 21.8, 22.7, 23.6, 22.1, 0.5, 0.866],
                [25.2, 21.0, 21.9, 22.8, 23.7, 22.2, 0.5, 0.866],
                [25.3, 21.1, 22.0, 22.9, 23.8, 22.3, 0.5, 0.866],
                [25.4, 21.2, 22.1, 23.0, 23.9, 22.4, 0.5, 0.866],
                [25.5, 21.3, 22.2, 23.1, 24.0, 22.5, 0.5, 0.866],
                [25.6, 21.4, 22.3, 23.2, 24.1, 22.6, 0.5, 0.866]
            ]
        ]
    )


class PredictionResponse(BaseModel):
    """Respuesta con predicciones de cuantiles"""
    quantiles: List[float] = Field(description="Valores de cuantiles")
    predictions: List[List[float]] = Field(
        description="Predicciones por cuantil (shape: decoder_length x num_quantiles)"
    )
    median: List[float] = Field(description="Mediana de las predicciones")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cargar modelo al iniciar y liberar recursos al cerrar"""
    
    try:
        logger.info("Inicializando API...")
        config = load_config("config_lstm.yaml")
        
        model_name = config["model_to_train"]
        
        # Construir modelo
        model_config = config["model_configs"][model_name]
        model_kwargs = {
            "input_size": len(config["features"]),
            "decoder_length": config["decoder_length"],
            "quantiles": config["quantiles"],
            **model_config,
        }

        model = LSTMModel(**model_kwargs)
        
        # Cargar checkpoint
        checkpoint_dir = Path(get_project_root()) / config["checkpoint_dir"] / model_name
        if not checkpoint_dir.exists():
            logger.error(f"Directorio de checkpoints no encontrado: {checkpoint_dir}")
            raise FileNotFoundError(f"Directorio de checkpoints no encontrado: {checkpoint_dir}")
        
        # Buscar el checkpoint más reciente
        checkpoints = list(checkpoint_dir.glob("**/*.ckpt"))
        if not checkpoints:
            logger.error(f"No hay checkpoints en {checkpoint_dir}")
            raise FileNotFoundError(f"No hay checkpoints en {checkpoint_dir}")
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Cargando checkpoint: {latest_checkpoint}")

        model = LSTMModel.load_from_checkpoint(str(latest_checkpoint), **model_kwargs)
        
        model.eval()
        
        # Almacenar en estado de la app
        app.state.model = model
        app.state.config = config
        app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(app.state.device)
        
        logger.info(f"API inicializada correctamente. Modelo: {model_name}")
        
    except Exception as e:
        logger.error(f"Error al inicializar la API: {str(e)}")
        raise
    
    yield
    
    logger.info("Cerrando API...")


app = FastAPI(
    title="API de Inferencia para Series Temporales",
    description="API para realizar predicciones de series temporales con modelos Deep Learning",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    """Ruta raíz para confirmar que la API está disponible"""
    return {
        "status": "ok",
        "message": "API de inferencia activa",
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(encoder_input: EncoderInput):
    """Realizar predicción de serie temporal"""
    try:
        logger.info("Nueva solicitud de predicción recibida")
        
        # Validar entrada
        encoder_data = np.array(encoder_input.features, dtype=np.float32)
        
        config = app.state.config
        expected_features = len(config["features"])
        expected_length = config["encoder_length"]
        
        if encoder_data.shape[0] != expected_length:
            raise ValueError(
                f"Longitud del encoder incorrecta. "
                f"Esperado: {expected_length}, Recibido: {encoder_data.shape[0]}"
            )
        
        if encoder_data.shape[1] != expected_features:
            raise ValueError(
                f"Número de features incorrecto. "
                f"Esperado: {expected_features}, Recibido: {encoder_data.shape[1]}"
            )
        
        # Convertir a tensor
        encoder_tensor = torch.tensor(
            encoder_data.reshape(1, expected_length, expected_features),
            dtype=torch.float32
        ).to(app.state.device)
        
        # Decoder input: usar zeros con la misma forma
        decoder_tensor = torch.zeros(
            1, config["decoder_length"], expected_features,
            dtype=torch.float32
        ).to(app.state.device)
        
        # Predicción
        with torch.no_grad():
            predictions = app.state.model(encoder_tensor, decoder_tensor)
            predictions = predictions.cpu().numpy()
        
        # Extraer resultados
        predictions = predictions[0]  # (decoder_length, num_quantiles)
        quantiles = config["quantiles"]
        median_idx = quantiles.index(0.5)
        median = predictions[:, median_idx].tolist()
        
        logger.info(f"Predicción realizada exitosamente")
        
        return PredictionResponse(
            quantiles=quantiles,
            predictions=predictions.tolist(),
            median=median
        )
        
    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error realizando predicción: {str(e)}")


# ============================================================================
# Instrucciones de uso
# ============================================================================

"""
CÓMO EJECUTAR LA API:

1. Instalar dependencias (si no están ya instaladas):
   pip install fastapi uvicorn

2. Ejecutar la API:
   uvicorn src.api_inference:app --reload --host 0.0.0.0 --port 8000

3. Acceder a la documentación interactiva:
   http://localhost:8000/docs

4. Ejemplo de solicitud POST a /predict:
{
    "features": [
      [22.5, 18.3, 19.2, 20.1, 21.0, 19.5, 0.5, 0.866],
      [22.6, 18.4, 19.3, 20.2, 21.1, 19.6, 0.5, 0.866],
      [22.7, 18.5, 19.4, 20.3, 21.2, 19.7, 0.5, 0.866],
      [22.8, 18.6, 19.5, 20.4, 21.3, 19.8, 0.5, 0.866],
      [22.9, 18.7, 19.6, 20.5, 21.4, 19.9, 0.5, 0.866],
      [23.0, 18.8, 19.7, 20.6, 21.5, 20.0, 0.5, 0.866],
      [23.1, 18.9, 19.8, 20.7, 21.6, 20.1, 0.5, 0.866],
      [23.2, 19.0, 19.9, 20.8, 21.7, 20.2, 0.5, 0.866],
      [23.3, 19.1, 20.0, 20.9, 21.8, 20.3, 0.5, 0.866],
      [23.4, 19.2, 20.1, 21.0, 21.9, 20.4, 0.5, 0.866],
      [23.5, 19.3, 20.2, 21.1, 22.0, 20.5, 0.5, 0.866],
      [23.6, 19.4, 20.3, 21.2, 22.1, 20.6, 0.5, 0.866],
      [23.7, 19.5, 20.4, 21.3, 22.2, 20.7, 0.5, 0.866],
      [23.8, 19.6, 20.5, 21.4, 22.3, 20.8, 0.5, 0.866],
      [23.9, 19.7, 20.6, 21.5, 22.4, 20.9, 0.5, 0.866],
      [24.0, 19.8, 20.7, 21.6, 22.5, 21.0, 0.5, 0.866],
      [24.1, 19.9, 20.8, 21.7, 22.6, 21.1, 0.5, 0.866],
      [24.2, 20.0, 20.9, 21.8, 22.7, 21.2, 0.5, 0.866],
      [24.3, 20.1, 21.0, 21.9, 22.8, 21.3, 0.5, 0.866],
      [24.4, 20.2, 21.1, 22.0, 22.9, 21.4, 0.5, 0.866],
      [24.5, 20.3, 21.2, 22.1, 23.0, 21.5, 0.5, 0.866],
      [24.6, 20.4, 21.3, 22.2, 23.1, 21.6, 0.5, 0.866],
      [24.7, 20.5, 21.4, 22.3, 23.2, 21.7, 0.5, 0.866],
      [24.8, 20.6, 21.5, 22.4, 23.3, 21.8, 0.5, 0.866],
      [24.9, 20.7, 21.6, 22.5, 23.4, 21.9, 0.5, 0.866],
      [25.0, 20.8, 21.7, 22.6, 23.5, 22.0, 0.5, 0.866],
      [25.1, 20.9, 21.8, 22.7, 23.6, 22.1, 0.5, 0.866],
      [25.2, 21.0, 21.9, 22.8, 23.7, 22.2, 0.5, 0.866],
      [25.3, 21.1, 22.0, 22.9, 23.8, 22.3, 0.5, 0.866],
      [25.4, 21.2, 22.1, 23.0, 23.9, 22.4, 0.5, 0.866],
      [25.5, 21.3, 22.2, 23.1, 24.0, 22.5, 0.5, 0.866],
      [25.6, 21.4, 22.3, 23.2, 24.1, 22.6, 0.5, 0.866]
    ]
}

5. Autenticación con Weights & Biases (opcional):
   wandb login
"""
