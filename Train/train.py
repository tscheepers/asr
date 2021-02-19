#!/usr/bin/python

import sys
import pytorch_lightning
from qualitative_evaluation import QualitativeEvaluationCallback
from model import Model


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

    pytorch_lightning.Trainer(
        max_epochs=1000, gpus=1,
        gradient_clip_val=400,
        progress_bar_refresh_rate=5,
        resume_from_checkpoint=checkpoint,
        val_check_interval=2500,
        weights_summary='full',
        callbacks=[
            QualitativeEvaluationCallback(),
            pytorch_lightning.callbacks.ModelCheckpoint(
                dirpath=output_dir, monitor='val_wer',
                save_top_k=5, save_last=True, mode='min'
            )
        ],
        logger=pytorch_lightning.loggers.TensorBoardLogger('./logs', name='speech_recognition')
    ).fit(model)


if __name__ == '__main__':
    main(sys.argv[1:])
