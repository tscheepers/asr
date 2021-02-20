import torch
from data.generate_spectrogram import generate_spectrogram
from data.string_processor import StringProcessor
from lib.spec_augment import spec_augment
from config import Config
from dataclasses import dataclass


@dataclass
class Sample:
    sentence: str
    path: str


class Dataset(torch.utils.data.Dataset):

    def __init__(self, file: str, string_processor: StringProcessor, config: Config, train: bool = False):
        super(Dataset, self).__init__()

        self.samples = []
        with open(file, 'r') as r:
            for line in r.readlines():
                (sentence, path) = line.split('\t')
                self.samples.append(Sample(sentence, path))

        self.train = train
        self.config = config
        self.string_processor = string_processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self._getitem(idx)

    def _getitem(self, idx):
        sample = self.samples[idx]

        labels = torch.Tensor(self.string_processor.str_to_labels(sample.sentence))
        n_labels = len(labels)

        if n_labels == 0:
            raise Exception('Zero labels')

        spectrogram = torch.Tensor(generate_spectrogram(sample.path, self.config))
        n_timesteps = spectrogram.shape[-1]

        if self.train and self.config.spec_augment:
            spectrogram = spec_augment(spectrogram)

        if n_timesteps > self.max_timesteps:
            raise Exception('Spectrogram has too many timesteps, size %s' % n_timesteps)

        if n_timesteps < n_labels:
            raise Exception('Less timesteps than labels')

        return spectrogram, labels, n_timesteps, n_labels


def collate_dataset(dataset):
    """
    Generate a padded training batch from a dataset
    """
    spectrograms = []
    labels = []
    n_timesteps = []
    n_labels = []

    # Order by number of timesteps descending
    for (s, l, ss, ls) in sorted(dataset, key=lambda x: -x[2]):
        spectrograms.append(s.transpose(0, 1))
        labels.append(l)
        n_timesteps.append(ss)
        n_labels.append(ls)

    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).transpose(1, 2)  # batch, spect, time
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)  # batch, time
    n_timesteps = torch.IntTensor(n_timesteps)  # batch
    n_labels = torch.IntTensor(n_labels)  # batch

    return spectrograms, labels, n_timesteps, n_labels
