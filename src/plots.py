"""Módulo para graficar predicciones de modelos de series temporales con bandas de confianza basadas en cuantiles"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _save_figure(fig, save_path):
    """Guardar figura en la ruta especificada, creando directorios si es necesario"""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    return str(save_path)


def plot_prediction(y_true, y_pred, quantiles, title="Predicción", save_path=None, n_steps=250):
    """Graficar predicción con bandas de confianza basadas en cuantiles"""
    
    # Aplanar arrays si están en formato de múltiples muestras
    if y_true.ndim == 2:
        y_true_flat = y_true.flatten()
    else:
        y_true_flat = y_true
    
    if y_pred.ndim == 3:
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        y_pred_flat = y_pred
    
    # Limitar al número de pasos especificado
    if n_steps is not None:
        y_true_flat = y_true_flat[:n_steps]
        y_pred_flat = y_pred_flat[:n_steps]
    
    fig = plt.figure(figsize=(14, 4))
    time_steps = np.arange(len(y_true_flat))
    
    # Graficar valor real
    plt.plot(time_steps, y_true_flat, 'k-', linewidth=2, label='Real')
    
    # Graficar mediana
    median_idx = quantiles.index(0.5)
    plt.plot(time_steps, y_pred_flat[:, median_idx], 'b-', linewidth=1.5, label='Mediana (Predicción)')
    
    # Graficar bandas de confianza
    q_low_idx = quantiles.index(0.1)
    q_high_idx = quantiles.index(0.95)
    plt.fill_between(time_steps, y_pred_flat[:, q_low_idx], y_pred_flat[:, q_high_idx], 
                     alpha=0.2, color='blue', label='10%-90% (Confianza)')
    
    plt.xlabel('Tiempo')
    plt.ylabel('Temperatura')
    plt.title(title)
    plt.legend(loc='upper right')

    if save_path:
        return _save_figure(fig, save_path)

    plt.show()
    return fig
