import librosa
import numpy as np
import torch
from ctcdecode import CTCBeamDecoder

from config import Config
from qualitative_evaluation import QualitativeEvaluator
from model import Model


def sample(model):
    sentence = "in eighteen sixty two a law was enacted with the purpose of suppressing plural marriage and as had " \
               "been predicted in the national senate prior to its passage it lay for many years a dead letter"
    labels = model.string_processor.str_to_labels(sentence)

    wave, sample_rate = librosa.load('../Tests/Fixtures/librispeech-sample.wav', sr=model.config.sample_rate, mono=True)
    stft = librosa.stft(wave,
                        n_fft=int(model.config.sample_rate * model.config.window_size),
                        hop_length=int(model.config.sample_rate * model.config.window_stride),
                        win_length=int(model.config.sample_rate * model.config.window_size),
                        window='hamming')
    magnitudes, _ = librosa.magphase(stft)
    spectrogram = np.log1p(magnitudes)
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

    return spectrogram, labels


if __name__ == '__main__':
    model = Model.load_from_checkpoint(
        './checkpoints-long-run-librispeech-without-spec-augment/epoch=11-step=208357.ckpt',
        config=Config(time_padding=True)
    )

    evaluator = QualitativeEvaluator(
        model,
        model.string_processor,
        CTCBeamDecoder(
            model.string_processor.chars,
            beam_width=100,
            blank_id=model.string_processor.blank_id,
            log_probs_input=True
        )
    )

    spectrogram, labels = sample(model)

    features = spectrogram.shape[0]
    timesteps = spectrogram.shape[1]

    timesteps_per_iteration = 50
    padding = 15
    frame_width = timesteps_per_iteration + 2 * padding
    iterations = (timesteps // timesteps_per_iteration)

    lstm_hn = torch.zeros(model.config.num_layers, model.config.hidden_size, device=model.device)
    lstm_cn = torch.zeros(model.config.num_layers, model.config.hidden_size, device=model.device)
    ys = None

    for i in range(iterations):
        start = i * timesteps_per_iteration - padding
        end = (i+1) * timesteps_per_iteration + padding

        if i == 0:
            slice = spectrogram[:,:end]
            iter_spectrogram = np.zeros((features, frame_width))
            iter_spectrogram[:,-slice.shape[1]:] = slice

        elif i == iterations - 1:
            slice = spectrogram[:, start:(i+1) * timesteps_per_iteration]
            iter_spectrogram = np.zeros((features, frame_width))
            iter_spectrogram[:, :slice.shape[1]] = slice
        else:
            iter_spectrogram = spectrogram[:, start:end]

        iter_spectrogram = torch.Tensor(iter_spectrogram, device=model.device)
        y, lstm_hn, lstm_cn = model.forward(iter_spectrogram, lstm_h0=lstm_hn, lstm_c0=lstm_cn)

        if ys is None:
            ys = y
        else:
            ys = torch.cat((ys, y))

    print("Streaming")
    evaluator.print_evaluation_of_output(ys, labels)

    print("Regular")
    evaluator.print_evaluation_of_sample(spectrogram, labels)
