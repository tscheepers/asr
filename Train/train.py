#!/usr/bin/python

import sys
import pytorch_lightning
from dataclasses import dataclass
from qualitative_evaluation import QualitativeEvaluationCallback
from model.cnn_rnn_lookahead_acoustic_model import CnnRnnLookaheadAcousticModel as Model


@dataclass
class TrainConfig:
    max_epochs: int = 200
    gradient_clip_val: int = 400
    val_check_interval: int = 2500


def main(args):

    if len(args) == 0:
        print('Please use as follows: train.py <checkpoint_output_directory> [<checkpoint_to_load.ckpt>]')
        sys.exit(2)

    output_dir = args[0]
    checkpoint = None if len(args) >= 1 else args[1]

    if checkpoint is not None:
        model = Model.load_from_checkpoint(checkpoint)
    else:
        model = Model()

    train_config = TrainConfig()
    pytorch_lightning.Trainer(
        max_epochs=train_config.max_epochs,
        gradient_clip_val=train_config.gradient_clip_val,
        gpus=1,
        resume_from_checkpoint=checkpoint,
        val_check_interval=train_config.val_check_interval,
        weights_summary='full',
        callbacks=[
            QualitativeEvaluationCallback(),
            pytorch_lightning.callbacks.ModelCheckpoint(
                dirpath=output_dir, monitor='val_wer',
                save_top_k=5, save_last=True, mode='min'
            )
        ],
        logger=pytorch_lightning.loggers.TensorBoardLogger('./logs', name='cnn_rnn_lookahead_acoustic_model')
    ).fit(model)


if __name__ == '__main__':
    main(sys.argv[1:])
