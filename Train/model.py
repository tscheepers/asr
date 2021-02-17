import pytorch_lightning
import torch

from config import Config
from dataset import LibriSpeechDataset, collate_dataset, StringProcessor
from decoder import GreedyDecoder
from layer import MaskConv, SequenceWise, BatchLSTM, Lookahead
from validation import WordErrorRate, CharErrorRate


class Model(pytorch_lightning.core.lightning.LightningModule):

    def __init__(self, config=Config()):
        super(Model, self).__init__()

        self.config = config
        self.string_processor = StringProcessor()

        # STFT outputs nFFT / 2 + 1 features
        input_size = int(config.sample_rate * config.window_size) // 2 + 1

        # When training we should use zero padding on the time dimension
        # When performing streaming inference we should not
        time_padding = 5 if self.config.time_padding else 0

        self.conv = MaskConv(torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, time_padding)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0.0, 20.0, inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, time_padding)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0.0, 20.0, inplace=True)
        ))

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = (input_size + 2 * 20 - 41) // 2 + 1
        rnn_input_size = (rnn_input_size + 2 * 10 - 21) // 2 + 1
        rnn_input_size *= 32

        self.rnns = torch.nn.Sequential(
            BatchLSTM(
                input_size=rnn_input_size,
                hidden_size=config.hidden_size,
                batch_norm=False
            ),
            *(
                BatchLSTM(
                    input_size=config.hidden_size,
                    hidden_size=config.hidden_size,
                ) for _ in range(config.num_layers - 1)
            )
        )

        self.lookahead = torch.nn.Sequential(
            Lookahead(config.hidden_size, context=config.lookahead_context),
            torch.nn.Hardtanh(0.0, 20.0, inplace=True)
        )

        self.fc = SequenceWise(torch.nn.Sequential(
            torch.nn.BatchNorm1d(config.hidden_size),
            torch.nn.Linear(config.hidden_size, len(self.string_processor.chars), bias=False)
        ))

        self.criterion = torch.nn.CTCLoss(blank=self.string_processor.blank_id, reduction='sum', zero_infinity=True)

        self.evaluation_decoder = GreedyDecoder(
            self.string_processor.chars,
            blank_index=self.string_processor.blank_id
        )  # Decoder used for validation

        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )

        self.train_dataset = LibriSpeechDataset(
            self.string_processor, config, spec_augment=True,
            filepath='/home/thijs/Datasets/LibriSpeech/train_transcriptions.tsv'
        )
        self.test_dataset = LibriSpeechDataset(
            self.string_processor, config,
            filepath='/home/thijs/Datasets/LibriSpeech/test_transcriptions.tsv'
        )
        self.val_dataset = LibriSpeechDataset(
            self.string_processor, config,
            filepath='/home/thijs/Datasets/LibriSpeech/dev_transcriptions.tsv'
        )

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
            betas=self.config.adam_betas,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.config.lr_gamma
        )

        return [optimizer], [scheduler]

    def forward(self, spectrograms, lstm_h0=None, lstm_c0=None, n_timesteps=None):
        x = spectrograms

        # Allow for single value inputs
        if len(spectrograms.shape) == 2:
            x = spectrograms.unsqueeze(0) # batch, feature, time

        n = self.get_n_hidden(n_timesteps) if n_timesteps is not None else None  # batch
        x = x.unsqueeze(1) # batch, channel, feature, time
        x, _ = self.conv(x, n)

        batch_size, channels, features, timesteps = x.size()
        x = x.view(batch_size, channels * features, timesteps)  # collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # time, batch, feature

        if lstm_h0 is None:
            lstm_h0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=self.device)
        elif len(lstm_h0.shape) == 2:
            lstm_h0 = lstm_h0.unsqueeze(1)

        if lstm_c0 is None:
            lstm_c0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=self.device)
        elif len(lstm_c0.shape) == 2:
            lstm_c0 = lstm_c0.unsqueeze(1)

        lstm_hn = lstm_cn = None
        for i, lstm in enumerate(self.rnns):
            x, hn, cn = lstm(x, lstm_h0[i,:,:].unsqueeze(0), lstm_c0[i,:,:].unsqueeze(0), n)  # time, batch, feature
            lstm_hn = hn if lstm_hn is None else torch.cat((lstm_hn, hn))
            lstm_cn = cn if lstm_cn is None else torch.cat((lstm_cn, cn))

        x = self.lookahead(x)  # time, batch, feature
        x = self.fc(x)  # time, batch, label
        x = x.transpose(0, 1)  # batch, time, label
        x = x.log_softmax(-1)  # batch, time, label

        # Return single value inputs
        if len(spectrograms.shape) == 2:
            x = x.squeeze(0)
            lstm_hn = lstm_hn.squeeze(1)
            lstm_cn = lstm_cn.squeeze(1)

        # When n_timesteps is not passed, we are not using padding and masking
        if n_timesteps is not None:
            return x, n, lstm_hn, lstm_cn

        return x, lstm_hn, lstm_cn

    def get_n_hidden(self, n_timesteps):
        n = n_timesteps.cpu().int()
        for m in self.conv.modules():
            if type(m) == torch.nn.modules.conv.Conv2d:
                n = ((n + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return n.int()

    def training_step(self, batch, batch_idx):
        spectrograms, labels, n_timesteps, n_labels = batch
        y, n_y, _, _ = self.forward(spectrograms, n_timesteps=n_timesteps)
        loss = self.criterion(y.transpose(0, 1), labels, n_y, n_labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_dataset,
            num_workers=self.config.num_workers
        )

    def validation_step(self, batch, batch_idx):
        spectrograms, labels, n_timesteps, n_labels = batch
        y, n_y, _, _ = self.forward(spectrograms, n_timesteps=n_timesteps)
        loss = self.criterion(y.transpose(0, 1), labels, n_y, n_labels)

        self.wer(
            preds=y,
            preds_sizes=n_y,
            targets=labels,
            target_sizes=n_labels
        )

        self.cer(
            preds=y,
            preds_sizes=n_y,
            targets=labels,
            target_sizes=n_labels
        )

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_wer', self.wer.compute(), prog_bar=True, on_epoch=True, logger=True)
        self.log('val_cer', self.cer.compute(), prog_bar=True, on_epoch=True, logger=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_dataset,
            num_workers=self.config.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_dataset,
            num_workers=self.config.num_workers
        )
