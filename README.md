# Deep Learning para series temporales - MLOps

Predicción de series temporales con distintas arquitecturas de redes neuronales (MLP, LSTM, GRU, TCN, Transformer) usando PyTorch Lightning y W&B.

**Autor:** Sergio Escudero Medina

**Enlaces del proyecto:**
- GitHub: https://github.com/sqdro/mlops-dl-timeseries
- Weights & Biases: https://wandb.ai/sergio-escudero-upm/TimeSeriesForecasting
  - Report W&B (Entrenamiento): https://api.wandb.ai/links/sergio-escudero-upm/4wul7jrs
  - Report W&B (Prediccciones): https://wandb.ai/sergio-escudero-upm/TimeSeriesForecasting/reports/Predicciones-de-la-serie-por-modelo--VmlldzoxNjgyNTc1NA?accessToken=a89xy08jsc7vdijftd3kdrryu7v0nd5ww2qmvwuv940dtn6sy4w5jge96pa6d85p

---

### 1. Entrenar un modelo

```bash
python -m src.main --config config/config.yaml
```

O cualquier configuración disponible en `config/` para los distintos modelos entrenados:
- `config/config_lstm.yaml`
- `config/config_gru.yaml`
- `config/config_mlp.yaml`
- `config/config_tcn.yaml`
- `config/config_transformer.yaml`

### 2. API de inferencia

```bash
uvicorn src.api_inference:app --reload
```

Acceder a la documentación interactiva en:

```
http://localhost:8000/docs
```

## 3. Dockerización

El proyecto se puede ejecutar con el `Dockerfile` del repositorio.

### Build de la imagen

```bash
docker build -t dl-timeseries .
```

### Ejecutar la API

```bash
docker run --rm -p 8000:8000 -v "%cd%/checkpoints:/app/checkpoints" dl-timeseries
```

### Ejecutar entrenamiento dentro del contenedor

```bash
docker run --rm \
  -v "%cd%/config:/app/config" \
  -v "%cd%/data:/app/data" \
  -v "%cd%/checkpoints:/app/checkpoints" \
  -v "%cd%/logs:/app/logs" \
  dl-timeseries -m src.main --config config/config_lstm.yaml
```

## 4. Tests

```bash
pytest tests/ -v
```

Tests disponibles:
- `tests/conftest.py`: fixture y configuración general compartida en los diferentes tests
- `tests/test_data.py`: tests sobre el conjunto de datos y su procesamiento
- `tests/test_models.py`: tests sobre los distintos modelos y su estructura

---

## Estructura del Proyecto

- Código original en `notebook/`.
- Carpeta `src/` contiene:
  - `main.py`: módulo principal para ejecutar el pipeline del entrenamiento del modelo elegido.
  - `train.py`, `losses.py`, `predict.py`, `plots.py`: módulos con el entrenamiento, evaluación y predicción del modelo elegido.
  - `utils.py`: módulo con funciones generales para los distintos módulos.
  - `api_inference.py`: módulo con la configuración de la API de inferencia.
- Carpeta `data/` contiene el conjunto de datos en formato CSV y el módulo del procesamiento de este.
- Carpeta `models/` contiene la estructura de cada modelo.
- Carpeta `config/` contiene las rutas e hiperparámetros del proyecto, así como los parámetros de cada modelo.
- Carpeta `tests/` contiene los módulos con los test a ejecutar.

```
.
├── notebooks/
|   └── main.ipynb
├── src/
│   ├── api_inference.py
│   ├── losses.py
│   ├── main.py
│   ├── plots.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── data/
│   ├── data_treatment.py
│   └── train.csv
├── models/
│   ├── gru.py
│   ├── lstm.py
│   ├── mlp.py
│   ├── tcn.py
│   └── transformer.py
├── tests/
|   ├── conftest.py
│   ├── test_data.py
│   └── test_models.py
├── config/
│   ├── config_gru.yaml
│   ├── config_lstm.yaml
│   ├── config_mlp.yaml
│   ├── config_tcn.yaml
│   └── config_transformer.yaml
├── checkpoints/
├── logs/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Modelos disponibles

| Modelo | Arquitectura básica |
|--------|---------------------|
| **MLP** | Multilayer Perceptron |
| **LSTM** | Long Short-Term Memory |
| **GRU** | Gated Recurrent Unit |
| **TCN** | Temporal Convolutional Network |
| **Transformer** | Transformer |

---

## Configuración

Los archivos de configuración están en `config/`. Cada uno define las rutas del proyecto, la lista de `features` y `target` y los distintos hiperparámetros del proyecto y parámetros por modelo.

### Valores clave

- `data_path`: ruta al CSV de entrenamiento
- `features`: lista de características de entrada
- `target`: nombre de la variable objetivo
- `encoder_length`: longitud en el encoder (si aplica en el modelo)
- `decoder_length`: longitud en el decoder
- `batch_size`: tamaño de batch
- `quantiles`: cuantiles a predecir
- `model_to_train`: modelo a entrenar (a elegir entre `mlp`, `lstm`, `gru`, `tcn` y `transformer`)

---

## Weights & Biases (W&B)

El entrenamiento registra métricas y artefactos en W&B.

Métricas registradas:
- Pérdida en entrenamiento por epoch
- Pérdida en validación por epoch

Artefactos registrados:
- Checkpoint del modelo entrenado
- Gráfico de la predicción final del modelo de los `n_steps` primeros pasos de la serie

---

## API de inferencia

### Endpoints disponibles

#### `POST /predict`

Envía una matriz de valores históricos para obtener predicciones por cuantil.

Ejemplo de entrada:

```bash
  '{
    "features": [
      [22.5, 18.3, 19.2, 20.1, 21.0, 19.5, 0.5, 0.866],
      [23.1, 18.9, 19.8, 20.7, 21.6, 20.1, 0.5, 0.866],
      ...
    ]
  }'
```

Ejemplo de salida:

```bash
  '{
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.95],
    "predictions": [
      [0.35, 0.40, 0.43, 0.59, 1.13],
      [0.45, 0.43, 0.43, 0.45, 1.05],
      [0.36, 0.34, 0.40, 0.52, 1.12],
      [0.31, 0.35, 0.38, 0.55, 1.00],
      [0.42, 0.41, 0.44, 0.49, 1.15],
      [0.38, 0.50, 0.50, 0.60, 1.16],
      [0.35, 0.42, 0.43, 0.58, 1.13]
    "median": [0.43, 0.45, 0.49, 0.51, 0.49, 0.48, 0.49]
  }'
```

- `features`: debe tener `encoder_length` filas y el mismo número de columnas que `features` en la configuración.

---

## Logs

- **Training logs**: `logs/model.log`
- **API logs**: `logs/api.log`
