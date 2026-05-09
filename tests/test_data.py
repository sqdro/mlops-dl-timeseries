"""Tests para el módulo de carga y procesamiento de datos"""

import pytest
import numpy as np
import pandas as pd
import torch

from data.data_treatment import TimeSeriesDataset, load_dataset, split_and_scale_data, build_dataloaders
from src.utils import load_config


class TestTimeSeriesDataset:
    """Tests para la clase TimeSeriesDataset"""
    
    def test_dataset_creation(self, sample_dataframe, dataset_params):
        """Test que el dataset se inicializa correctamente"""

        dataset = TimeSeriesDataset(sample_dataframe, **dataset_params)
        
        assert dataset.encoder_length == 32
        assert dataset.decoder_length == 7
        assert dataset.X.shape[1] == 8
        assert dataset.y.shape[0] == 200
    
    def test_dataset_length(self, sample_dataframe, dataset_params):
        """Test que la longitud del dataset es correcta"""

        dataset = TimeSeriesDataset(sample_dataframe, **dataset_params)
        
        expected_len = len(sample_dataframe) - 32 - 7 + 1
        assert len(dataset) == expected_len
    
    def test_dataset_values_are_float(self, sample_dataframe, dataset_params):
        """Test que todos los tensores son float"""

        dataset = TimeSeriesDataset(sample_dataframe, **dataset_params)
        item = dataset[0]
        
        assert item['encoder_x'].dtype == torch.float
        assert item['decoder_x'].dtype == torch.float
        assert item['decoder_y'].dtype == torch.float


class TestDataProcessing:
    """Tests para las funciones de procesamiento de datos"""
    
    @pytest.fixture
    def config_path(self):
        """Ruta al archivo de configuración"""
        return "config_lstm.yaml"
    
    def test_load_dataset(self, config_path):
        """Test que la carga del dataset se hace correctamente"""
        
        config = load_config(config_path)
        df = load_dataset(config["data_path"])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'sin_month' in df.columns
        assert 'cos_month' in df.columns
        assert df.isna().sum().sum() == 0  # Sin valores nulos
    
    def test_split_and_scale_data(self, sample_dataframe):
        """Test que la división y normalización de datos se hace correctamente"""

        features = ['feature_AA', 'feature_AB', 'feature_BA', 'feature_BB', 
                    'feature_CA', 'feature_CB', 'sin_month', 'cos_month']
        target = 'Temperature'
        
        df_train, df_val, scaler_X, scaler_y = split_and_scale_data(
            sample_dataframe,
            features=features,
            target=target,
            train_ratio=0.7
        )
        
        # Verificar split 70/30
        expected_train_size = int(200 * 0.7)
        assert len(df_train) == expected_train_size
        assert len(df_val) == 200 - expected_train_size
        
        # Verificar normalización (puede estar ligeramente fuera de [0,1] por la naturaleza de MinMaxScaler)
        for feat in features:
            assert df_train[feat].min() >= -0.2
            assert df_train[feat].max() <= 1.2
            assert df_val[feat].min() >= -0.2
            assert df_val[feat].max() <= 1.2
