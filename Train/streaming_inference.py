#!/usr/bin/python

import sys
import numpy as np
import torch
from data.generate_spectrogram import generate_spectrogram
from qualitative_evaluation import QualitativeEvaluator
from cnn_rnn_lookahead_acoustic_model import CnnRnnLookaheadAcousticModel as Model
from cnn_rnn_lookahead_acoustic_model import CnnRnnLookaheadAcousticModelConfig as Config

SAMPLE = (
    '../Tests/Fixtures/librispeech-sample.wav',
    'in eighteen sixty two a law was enacted with the purpose of suppressing plural marriage and as had ' \
    'been predicted in the national senate prior to its passage it lay for many years a dead letter'
)


def transform_into_spectrogram_labels(sample, model):
    filename, sentence = sample
    return generate_spectrogram(filename, model.data_config), model.string_processor.str_to_labels(sentence)


def inference_all_at_once(input_file):
    # Load model
    model = Model.load_from_checkpoint(input_file, model_config=Config(time_padding=True))

    # Set model to evaluation mode
    model.eval()
    model.zero_grad()
    torch.set_grad_enabled(False)

    # Create evaluator
    evaluator = QualitativeEvaluator(model)

    # Get example input
    spectrogram, labels = transform_into_spectrogram_labels(SAMPLE, model)

    # Print result from example input
    evaluator.print_evaluation_of_sample(spectrogram, labels, split_every=30)


def inference_in_chunks(input_file, useful_frame_width=60):
    # Load model, disable padding because we will pad the input ourselves
    model = Model.load_from_checkpoint(input_file, model_config=Config(time_padding=False))

    # Set model to evaluation mode
    model.eval()
    model.zero_grad()
    torch.set_grad_enabled(False)

    # Create evaluator
    evaluator = QualitativeEvaluator(model)

    # Get example input
    spectrogram, labels = transform_into_spectrogram_labels(SAMPLE, model)
    n_features, n_timesteps = spectrogram.shape

    padding = 15
    lookahead_overflow = model.model_config.lookahead_context * 2  # times two because of the strides in layer 1
    total_frame_width = padding + useful_frame_width + lookahead_overflow + padding  # add left and right padding
    iterations = (n_timesteps - lookahead_overflow) // useful_frame_width + 1
    useful_output_width = useful_frame_width // 2  # divide by two because of the strides in the first layer

    lstm_hn = torch.zeros(model.model_config.num_rnn_layers, model.model_config.hidden_size).to(model.device)
    lstm_cn = torch.zeros(model.model_config.num_rnn_layers, model.model_config.hidden_size).to(model.device)
    ys = None

    # Example below illustrates the chunking for 3 iterations:
    # -----|--------------------------------|-----
    #   0  | Entire input spectrogram       |  0
    # -----|--------|-----------------------|-----
    # | LP | UF     | LO  | RP |
    # -----|--------|--------|--------------------
    #          | LP | UF     | LO  | RP |
    # --------------|--------|--------------|-----
    #                   | LP | UF     | LO  | RP |
    # -----------------------|--------------|-----
    # PL = Left padding, RP = Right padding
    # UF = Useful frame, LO = Lookahead overflow
    #
    # Outputs from iterations:
    # -----|--------|--------|--------------|-----
    #      | 1      | 2      | 3            |
    # -----|--------|--------|--------------|-----
    for i in range(iterations):
        start = 0 if i == 0 else i * useful_frame_width - padding
        end = n_timesteps if i == iterations - 1 else (i+1) * useful_frame_width + lookahead_overflow + padding

        iter_start = padding if i == 0 else 0
        iter_end = iter_start + end - start

        iter_spectrogram = np.zeros((n_features, total_frame_width))
        iter_spectrogram[:, iter_start:iter_end] = spectrogram[:, start:end]
        iter_spectrogram = torch.Tensor(iter_spectrogram, device=model.device)

        # Pass in lstm_hn and lstm_cn from the previous iteration, so the hidden states are preserved
        y, lstm_hn, lstm_cn = model(iter_spectrogram, lstm_h0=lstm_hn, lstm_c0=lstm_cn)

        # Omit the output with lookahead overflow
        until = (end - start - (0 if i == 0 else 1) * padding) // 2 if i == iterations - 1 else useful_output_width
        y = y[:until]

        # Concatenate the final result
        ys = y if ys is None else torch.cat((ys, y))

    # Print result from example input
    evaluator.print_evaluation_of_output(ys, labels, split_every=useful_output_width)


def main(args):
    if len(args) != 1:
        print('Please provide a model checkpoint to use: streaming_inference.py <inputfile.ckpt>')
        sys.exit(2)

    checkpoint = args[0]

    print("Inference all at once (non-streaming):")
    inference_all_at_once(checkpoint)

    print('------------------------------------------')
    print("Inference in chunks (ready for streaming):")
    inference_in_chunks(checkpoint)


if __name__ == '__main__':
    main(sys.argv[1:])
