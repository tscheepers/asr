from dataclasses import dataclass


@dataclass
class DatasetConfig:
    batch_size: int = 16
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    max_timesteps: int = 4000
    spec_augment: bool = True
    num_workers: int = 16
    lowercase: bool = True
    valid_chars: str = ' \'abcdefghijklmnopqrstuvwxyz'
    blank_char: str = '.'
    train_file: str = '/home/thijs/Datasets/LibriSpeech/prepared_train_transcriptions.tsv'
    val_file: str = '/home/thijs/Datasets/LibriSpeech/prepared_dev_transcriptions.tsv'
    test_file: str = '/home/thijs/Datasets/LibriSpeech/prepared_test_transcriptions.tsv'