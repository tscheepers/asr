import librosa
import numpy as np
from .config import DatasetConfig


def generate_spectrogram(filename, config: DatasetConfig):
    wave, wave_sample_rate = librosa.load(filename, sr=config.sample_rate, mono=True)

    if config.sample_rate != wave_sample_rate:
        raise Exception('Sample rate mismatch %d != %d' % (wave_sample_rate, config.sample_rate))

    stft = librosa.stft(wave,
                        n_fft=int(config.sample_rate * config.window_size),
                        hop_length=int(config.sample_rate * config.window_stride),
                        win_length=int(config.sample_rate * config.window_size),
                        window='hamming')
    magnitudes, _ = librosa.magphase(stft)
    spectrogram = np.log1p(magnitudes)
    normalized_spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
    return normalized_spectrogram
