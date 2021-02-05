import math

import pytorch_lightning
import torch
from dataset import LibriSpeechDataset, pad_dataset, StringProcessor
from layer import MaskConv, SequenceWise, BatchRNN, Lookahead
from qualitative_evaluation import QualitativeEvaluation


class Model(pytorch_lightning.core.lightning.LightningModule):

    def __init__(self, string_processor: StringProcessor, batch_size=16, hidden_size=1024, num_classes=28,
                 n_features=161, num_layers=5, sample_rate=16000, window_size=0.02, window_stride=0.01,
                 max_timesteps=4000, lookahead_context=20):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.max_timesteps = max_timesteps
        self.n_features = n_features
        self.window_size = window_size
        self.window_stride = window_stride
        self.string_processor = string_processor
        self.lookahead_context = lookahead_context

        self.conv = MaskConv(torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0, 20, inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0, 20, inplace=True)
        ))

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor(self.sample_rate * self.window_size / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnns = torch.nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=self.hidden_size,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                ) for _ in range(self.num_layers - 1)
            )
        )

        self.lookahead = torch.nn.Sequential(
            Lookahead(self.hidden_size, context=self.lookahead_context),
            torch.nn.Hardtanh(0, 20, inplace=True)
        )

        self.fc = SequenceWise(torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Linear(self.hidden_size, num_classes, bias=False)
        ))

        self.criterion = torch.nn.CTCLoss(blank=self.string_processor.blank_id, zero_infinity=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=1e-3,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.99
        )

        return [optimizer], [scheduler]

    def forward(self, features, n_features):
        n = self.get_n_hidden(n_features)
        x, _ = self.conv(features.unsqueeze(1), n)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # time, batch, feature

        for rnn in self.rnns:
            x = rnn(x, n)

        x = self.lookahead(x)
        x = self.fc(x)

        return x, n

    def get_n_hidden(self, n_features):
        n = n_features.cpu().int()
        for m in self.conv.modules():
            if type(m) == torch.nn.modules.conv.Conv2d:
                n = ((n + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return n.int()

    def step(self, batch):
        features, labels, n_features, n_labels = batch
        y, n_y = self.forward(features, n_features)

        log_probabilities = torch.nn.functional.log_softmax(y, dim=2)
        loss = self.criterion(log_probabilities, labels, n_y, n_labels)

        return loss, log_probabilities

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch)
        return {'loss': loss}

    def train_dataloader(self):
        dataset = LibriSpeechDataset(
            self.string_processor,
            filepath='/home/thijs/Datasets/LibriSpeech/train-clean-100/transcripts.tsv',
            n_features=self.n_features,
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            window_stride=self.window_stride,
            max_timesteps=self.max_timesteps
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=pad_dataset,
            num_workers=16
        )

    def validation_step(self, batch, batch_idx):
        loss, _ = self.step(batch)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print('\n\n Total validation loss: %f' % avg_loss)

    def val_dataloader(self):
        dataset = LibriSpeechDataset(
            self.string_processor,
            filepath='/home/thijs/Datasets/LibriSpeech/dev-clean/transcripts.tsv',
            n_features=self.n_features,
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            window_stride=self.window_stride,
            max_timesteps=self.max_timesteps
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=pad_dataset,
            num_workers=16
        )

    def test_dataloader(self):
        dataset = LibriSpeechDataset(
            self.string_processor,
            filepath='/home/thijs/Datasets/LibriSpeech/test-clean/transcripts.tsv',
            n_features=self.n_features,
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            window_stride=self.window_stride,
            max_timesteps=self.max_timesteps
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=pad_dataset,
            num_workers=16
        )


if __name__ == '__main__':
    string_processor = StringProcessor()
    model = Model(string_processor)
    pytorch_lightning.Trainer(
        max_epochs=10000, gpus=1,
        num_nodes=1, distributed_backend=None,
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=5,
        callbacks=[QualitativeEvaluation()]
    ).fit(model)
