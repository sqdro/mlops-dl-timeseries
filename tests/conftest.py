"""Configuración compartida para todos los tests"""

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def random_seed():
    """Fijar seed para reproducibilidad en tests"""

    np.random.seed(42)
    torch.manual_seed(42)
    return 42


@pytest.fixture
def sample_dataframe(random_seed):
    """DataFrame de ejemplo para todos los tests de datos"""

    n_samples = 200
    features = ['feature_AA', 'feature_AB', 'feature_BA', 'feature_BB', 
                'feature_CA', 'feature_CB', 'sin_month', 'cos_month']
    target = 'Temperature'
    
    data = {f: np.random.randn(n_samples) for f in features}
    data[target] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def common_config():
    """Configuración común para tests de modelos y predicción"""

    return {
        'input_size': 8,
        'decoder_length': 7,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.95],
    }


@pytest.fixture
def batch_data(random_seed):
    """Batch de datos para tests de modelos"""

    batch_size = 4
    encoder_length = 32
    decoder_length = 7
    input_size = 8
    
    x_enc = torch.randn(batch_size, encoder_length, input_size)
    x_dec = torch.randn(batch_size, decoder_length, input_size)
    
    return x_enc, x_dec


@pytest.fixture
def dataset_params():
    """Parámetros para TimeSeriesDataset."""
    return {
        'features': ['feature_AA', 'feature_AB', 'feature_BA', 'feature_BB', 
                    'feature_CA', 'feature_CB', 'sin_month', 'cos_month'],
        'target': 'Temperature',
        'encoder_length': 32,
        'decoder_length': 7,
    }
