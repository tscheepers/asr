import pytorch_lightning
import torch

from qualitative_evaluation import QualitativeEvaluation
from model import Model


if __name__ == '__main__':
    model = Model()

    pytorch_lightning.Trainer(
        max_epochs=1000, gpus=1,
        gradient_clip_val=400,
        progress_bar_refresh_rate=5,
        # overfit_batches=1,
        # check_val_every_n_epoch=10,
        val_check_interval=5000,
        weights_summary='full',
        callbacks=[
            QualitativeEvaluation(),
            pytorch_lightning.callbacks.ModelCheckpoint(
                dirpath='./checkpoints', monitor='val_wer',
                save_top_k=1, save_last=True, mode='min'
            )
        ],
        logger=pytorch_lightning.loggers.TensorBoardLogger('./logs', name='speech_recognition')
    ).fit(model)
