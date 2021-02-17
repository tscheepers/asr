import pytorch_lightning
import torch

from qualitative_evaluation import QualitativeEvaluationCallback
from model import Model


if __name__ == '__main__':
    model = Model.load_from_checkpoint(
        './checkpoints-long-run-librispeech-without-spec-augment/epoch=11-step=208357.ckpt')

    pytorch_lightning.Trainer(
        max_epochs=1000, gpus=1,
        gradient_clip_val=400,
        progress_bar_refresh_rate=5,
        # overfit_batches=1,
        # check_val_every_n_epoch=10,
        val_check_interval=2500,
        weights_summary='full',
        callbacks=[
            QualitativeEvaluationCallback(),
            pytorch_lightning.callbacks.ModelCheckpoint(
                dirpath='./checkpoints-fine-tuning', monitor='val_wer',
                save_top_k=3, save_last=True, mode='min'
            )
        ],
        logger=pytorch_lightning.loggers.TensorBoardLogger('./logs', name='speech_recognition')
    ).fit(model)
