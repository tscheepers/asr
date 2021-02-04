import torch
import torchaudio
import re

from typing import List


class Sample:
    def __init__(self, sentence: str, path: str):
        self.path = path
        self.sentence = sentence

    def __repr__(self):
        return "%s [%s]" % (self.sentence, self.path)


class StringProcessor:
    def __init__(self):
        self.chars = [
            ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.'
        ]
        self.blank_id = self.chars.index('.')
        self.char_to_idx = {char: idx for (idx, char) in enumerate(self.chars)}

    def str_to_labels(self, string):
        # to lower case
        string = string.lower()

        # remove all chars not in the char list
        string = re.sub('[^a-z ]+', '', string)

        # remove double spaces
        string = re.sub(' +', ' ', string)

        # remove leading and trailing spaces
        string = string.strip()

        return [self.char_to_idx[char] for char in string]

    def labels_to_str(self, labels):
        return ''.join([self.chars[int(idx)] for idx in labels])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples: List[Sample], string_processor: StringProcessor, sample_rate=16000, n_features=64, max_timesteps=3000):
        super(Dataset, self).__init__()

        self.samples = samples
        self.max_timesteps = max_timesteps
        self.sample_rate = sample_rate

        self.audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_features
        )

        self.string_processor = string_processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        sample = self.samples[idx]

        labels = torch.Tensor(self.string_processor.str_to_labels(sample.sentence))
        n_labels = len(labels)

        if n_labels == 0:
            raise Exception('Zero labels')

        waveform, sample_rate = torchaudio.load(sample.path)

        if sample_rate < self.sample_rate:
            raise Exception('Sample rates mismatch %d < %d' % (sample_rate, self.sample_rate))

        features = self.audio_transform(waveform)

        if features.shape[0] > 1:
            raise Exception('Dual channel audio')

        features = features.reshape((features.shape[1], features.shape[2]))
        n_features = features.shape[-1]

        if n_features > self.max_timesteps:
            raise Exception('Spectrogram to big, size %s' % n_features)

        if n_features < n_labels:
            raise Exception('Less MFCCs than labels')

        return features, labels, n_features, n_labels


class CommonVoiceDataset(Dataset):
    def __init__(self, string_processor: StringProcessor, filename='dev.tsv', sample_rate=16000, n_features=64,
                 max_timesteps=3000, sample_limit=None):

        samples = []
        with open(CommonVoiceDataset.data_dir(filename), "r") as f:
            f.readline()  # Skip the first line
            lines = f.readlines()
            if sample_limit is not None:
                lines = lines[:sample_limit]

            for line in lines:
                split = line.split('\t')
                path = CommonVoiceDataset.data_dir('clips', split[1])
                sentence = split[2]
                samples.append(Sample(sentence, path))

        super(CommonVoiceDataset, self).__init__(samples, string_processor, sample_rate=sample_rate,
                                                 n_features=n_features, max_timesteps=max_timesteps)

    @staticmethod
    def data_dir(*args, root="/home/thijs/Datasets/cv-corpus-6.1-2020-12-11/nl"):
        return "/".join([root] + list(args))


class LibriSpeechDataset(Dataset):
    def __init__(self, string_processor: StringProcessor,
                 filepath='/home/thijs/Datasets/LibriSpeech/dev-clean/transcripts.tsv',
                 sample_rate=16000, n_features=64,
                 max_timesteps=3000, sample_limit=None):

        samples = []
        with open(filepath, "r") as f:
            f.readline()  # Skip the first line
            lines = f.readlines()
            if sample_limit is not None:
                lines = lines[:sample_limit]

            for line in lines:
                split = line.split('\t')
                path = split[0]
                sentence = split[2]
                samples.append(Sample(sentence, path))

        super(LibriSpeechDataset, self).__init__(samples, string_processor, sample_rate=sample_rate,
                                                 n_features=n_features, max_timesteps=max_timesteps)


def pad_dataset(dataset):
    all_features = []
    all_labels = []
    all_n_features = []
    all_n_labels = []

    for (features, labels, n_features, n_labels) in dataset:
        all_features.append(features.transpose(0, 1))
        all_labels.append(labels)
        all_n_features.append(n_features)
        all_n_labels.append(n_labels)

    all_features = torch.nn.utils.rnn.pad_sequence(all_features, batch_first=True).transpose(1, 2)
    all_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True)

    return all_features, all_labels, all_n_features, all_n_labels
