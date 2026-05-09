"""Módulo para entrenar modelos de series temporales con PyTorch Lightning"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from datetime import datetime
import wandb


def train_model(model, train_loader, val_loader, dir_checkpoint, 
    model_name='model', max_epochs=10, patience=5, min_delta=1e-4,
    gradient_clip_val=1.0, save_top_k=1
    ):
    """Entrenar modelo con PyTorch Lightning y callbacks de EarlyStopping y ModelCheckpoint"""

    if wandb.run is not None:
        logger_wandb = WandbLogger(experiment=wandb.run, log_model=model)
    else:
        logger_wandb = WandbLogger(project='TimeSeriesForecasting', name=model_name, log_model=model_name)

    # Crear directorio de checkpoints si no existe
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = Path(dir_checkpoint) / model_name / run_date
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        min_delta=min_delta
    )

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=save_top_k,
        filename=f'{model_name}-{{epoch:02d}}-{{val_loss:.2f}}',
        dirpath=str(checkpoint_path)
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        logger=logger_wandb,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    return model, trainer
