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


def regular():
    model = Model.load_from_checkpoint(
        './checkpoints-fine-tuning-padding-fix/epoch=12-step=223357.ckpt',
        config=Config(time_padding=True)
    )

    model.eval()
    model.zero_grad()
    torch.set_grad_enabled(False)

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
    evaluator.print_evaluation_of_sample(spectrogram[:,:1150], labels)


def streaming():
    model = Model.load_from_checkpoint(
        './checkpoints-fine-tuning-padding-fix/epoch=12-step=223357.ckpt',
        config=Config(time_padding=False)
    )

    model.eval()
    model.zero_grad()
    torch.set_grad_enabled(False)

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

    timesteps_per_iteration = 100
    lookahead = 40
    padding = 15
    frame_width = timesteps_per_iteration + 2 * padding
    iterations = (timesteps // (timesteps_per_iteration - lookahead))
    take_each_iteration = (timesteps_per_iteration - lookahead) // 2

    lstm_hn = torch.zeros(model.config.num_layers, model.config.hidden_size).to(model.device)
    lstm_cn = torch.zeros(model.config.num_layers, model.config.hidden_size).to(model.device)
    ys = None

    for i in range(iterations):
        start = i * (timesteps_per_iteration - lookahead) - padding
        offsetLeft = -min(start, 0)
        c_start = max(start, 0)

        end = (i + 1) * (timesteps_per_iteration - lookahead) + padding + lookahead
        offsetRight = max(iterations * timesteps_per_iteration, end) - iterations * timesteps_per_iteration
        c_end = min(end, iterations * timesteps_per_iteration)

        iter_spectrogram = np.zeros((features, frame_width))
        iter_spectrogram[:, offsetLeft:(timesteps_per_iteration + (padding - offsetRight) + offsetLeft + (padding - offsetLeft))] = spectrogram[:, c_start:c_end]
        iter_spectrogram = torch.Tensor(iter_spectrogram, device=model.device)

        y, lstm_hn, lstm_cn = model(iter_spectrogram, lstm_h0=lstm_hn, lstm_c0=lstm_cn)
        y = y[:take_each_iteration,:]
        ys = y if ys is None else torch.cat((ys, y))

    evaluator.print_evaluation_of_output(ys, labels, split_every=take_each_iteration)

    return ys.numpy()


if __name__ == '__main__':
    print("Regular")
    regular()
    print("Streaming")
    streaming()