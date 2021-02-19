import torch
from typing import List
from data.generate_spectrogram import generate_spectrogram
from data.string_processor import StringProcessor
from lib.spec_augment import spec_augment
from config import Config


class Sample:
    def __init__(self, sentence: str, path: str):
        self.path = path
        self.sentence = sentence

    def __repr__(self):
        return "%s [%s]" % (self.sentence, self.path)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples: List[Sample], string_processor: StringProcessor, config: Config, train: bool = False):
        super(Dataset, self).__init__()

        self.samples = samples
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

        if self.train and self.config.spec_augment:
            spectrogram = spec_augment(spectrogram)

        n_timesteps = spectrogram.shape[-1]

        if n_timesteps > self.max_timesteps:
            raise Exception('Spectrogram has too many timesteps, size %s' % n_timesteps)

        if n_timesteps < n_labels:
            raise Exception('Less timesteps than labels')

        return spectrogram, labels, n_timesteps, n_labels


class CommonVoiceDataset(Dataset):
    def __init__(self, string_processor: StringProcessor, config: Config, filename='dev.tsv', ):
        samples = []
        with open(CommonVoiceDataset.data_dir(filename), "r") as f:
            f.readline()  # Skip the first line
            for line in f.readlines():
                split = line.split('\t')
                path = CommonVoiceDataset.data_dir('clips', split[1])
                sentence = split[2]
                samples.append(Sample(sentence, path))

        super(CommonVoiceDataset, self).__init__(samples, string_processor, config)

    @staticmethod
    def data_dir(*args, root="/home/thijs/Datasets/cv-corpus-6.1-2020-12-11/nl"):
        return "/".join([root] + list(args))


class LibriSpeechDataset(Dataset):
    def __init__(self, string_processor: StringProcessor, config: Config,
                 filepath='/home/thijs/Datasets/LibriSpeech/dev_transcriptions.tsv', train: bool = False):
        samples = []
        with open(filepath, "r") as f:
            f.readline()  # Skip the first line
            for line in f.readlines():
                split = line.split('\t')
                path = split[0]
                sentence = split[2]
                samples.append(Sample(sentence, path))

        super(LibriSpeechDataset, self).__init__(samples, string_processor, config, train=train)


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
