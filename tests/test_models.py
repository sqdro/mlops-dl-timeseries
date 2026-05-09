"""Tests para los modelos a entrenar"""

import pytest
import torch
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.mlp import MLPModel
from models.tcn import TCNModel
from models.transformer import TransformerModel


class TestModelsOutput:
    """Tests de la salida de los modelos"""
    
    @pytest.mark.parametrize('model_class', [LSTMModel, GRUModel, MLPModel, TCNModel, TransformerModel])
    def test_output_shape(self, model_class, common_config, batch_data):
        """Test que todos los modelos devuelven la longitud correcta"""

        x_enc, x_dec = batch_data
        
        model_kwargs = {'hidden_size': 16, 'lr': 0.0001}
        if model_class == TransformerModel:
            model_kwargs.update({'num_heads': 1, 'num_layers': 1, 'encoder_length': 32})
        elif model_class not in (MLPModel, TCNModel):
            model_kwargs['num_layers'] = 1
        else:
            model_kwargs['encoder_length'] = 32
        
        model = model_class(**common_config, **model_kwargs)
        model.eval()
        
        with torch.no_grad():
            output = model(x_enc, x_dec)
        
        assert output.shape == (4, 7, 5) # batch_size x decoder_length x num_quantiles
    
    @pytest.mark.parametrize('model_class', [LSTMModel, GRUModel, MLPModel, TCNModel, TransformerModel])
    def test_output_no_nan_or_inf(self, model_class, common_config, batch_data):
        """Test que no hay NaN o Inf en las salidas de los modelos"""

        x_enc, x_dec = batch_data
        
        model_kwargs = {'hidden_size': 16, 'lr': 0.0001}
        if model_class == TransformerModel:
            model_kwargs.update({'num_heads': 1, 'num_layers': 1, 'encoder_length': 32})
        elif model_class not in (MLPModel, TCNModel):
            model_kwargs['num_layers'] = 1
        else:
            model_kwargs['encoder_length'] = 32
        
        model = model_class(**common_config, **model_kwargs)
        model.eval()
        
        with torch.no_grad():
            output = model(x_enc, x_dec)
        
        assert not torch.isnan(output).any(), f"NaN encontrado en {model_class.__name__}"
        assert not torch.isinf(output).any(), f"Inf encontrado en {model_class.__name__}"


class TestModelsInitialization:
    """Tests de inicialización de modelos"""

    def test_mlp_initialization(self, common_config):
        """Test inicialización del modelo MLP"""
        common_config.update({'encoder_length': 32})
        model = MLPModel(**common_config, hidden_size=16, lr=0.001)
        
        assert model.hidden_size == 16
        assert model.lr == 0.001

    def test_lstm_initialization(self, common_config):
        """Test inicialización del modelo LSTM"""
        model = LSTMModel(**common_config, hidden_size=16, num_layers=1, lr=0.001)
        
        assert model.hidden_size == 16
        assert model.num_layers == 1
        assert model.lr == 0.001
    
    def test_gru_initialization(self, common_config):
        """Test inicialización del modelo GRU"""
        model = GRUModel(**common_config, hidden_size=16, num_layers=1, lr=0.001)
        
        assert model.hidden_size == 16
        assert model.num_layers == 1
        assert model.lr == 0.001
    
    def test_tcn_initialization(self, common_config):
        """Test inicialización del modelo TCN"""
        common_config.update({'encoder_length': 32})
        model = TCNModel(**common_config, hidden_size=16, lr=0.001)
        
        assert model.hidden_size == 16
        assert model.lr == 0.001
    
    def test_transformer_initialization(self, common_config):
        """Test inicialización del modelo Transformer"""
        common_config.update({'encoder_length': 32})
        model = TransformerModel(**common_config, hidden_size=16, num_heads=1, num_layers=1, lr=0.001)
        
        assert model.hidden_size == 16
        assert model.num_heads == 1
        assert model.num_layers == 1
        assert model.lr == 0.001
