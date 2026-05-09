"""Módulo con funciones útiles para el proyecto de forecasting de series temporales"""

import yaml
import logging
from pathlib import Path


def get_project_root():
    """Obtener la ruta raíz del proyecto"""
    
    return Path(__file__).resolve().parent.parent


def setup_logger(name, level=logging.INFO, log_file="model.log"):
    """Configurar el logger para la aplicación"""

    logger_path = get_project_root() / "logs"
    logger_path.mkdir(exist_ok=True)
    logger_file = logger_path / log_file

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logger_file)
        ]
    )

    return logging.getLogger(name)


def load_config(config_name="config.yaml"):
    """Cargar archivo de configuración YAML desde carpeta config/"""

    config_path = get_project_root() / "config" / config_name

    if not config_path.exists():
        logger = logging.getLogger(__name__)
        logger.error(f"Archivo de configuración no encontrado: {config_path}")
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
