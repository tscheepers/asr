import pytorch_lightning
import torch
from config import Config
from dataset import LibriSpeechDataset, collate_dataset, StringProcessor
from lib.layers import MaskConv, SequenceWise, BatchLSTM, Lookahead
from lib.decoding import WordErrorRate, CharErrorRate, GreedyDecoder


class Model(pytorch_lightning.core.lightning.LightningModule):

    def __init__(self, config=Config(), string_processor=StringProcessor()):
        super(Model, self).__init__()

        self.config = config
        self.string_processor = string_processor

        # STFT outputs nFFT / 2 + 1 features
        input_size = int(config.sample_rate * config.window_size) // 2 + 1

        # When training we should use zero padding on the time dimension
        # When performing streaming inference we should not
        time_padding = 15 if self.config.time_padding else 0

        # Layer 1-2: 2 x Convolutional layers
        n_channels = config.conv_channels
        self.conv = MaskConv(torch.nn.Sequential(
            torch.nn.Conv2d(1, n_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, time_padding)),
            torch.nn.BatchNorm2d(n_channels),
            torch.nn.Hardtanh(config.tanh_min, config.tanh_max, inplace=True),
            torch.nn.Conv2d(n_channels, n_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 0)),
            torch.nn.BatchNorm2d(n_channels),
            torch.nn.Hardtanh(config.tanh_min, config.tanh_max, inplace=True)
        ))

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = (input_size + 2 * 20 - 41) // 2 + 1
        rnn_input_size = (rnn_input_size + 2 * 10 - 21) // 2 + 1
        rnn_input_size *= 32

        # Layer 3-8: 5 x LSTM Layers
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

        # Layer 9: Lookahead layer
        self.lookahead = torch.nn.Sequential(
            Lookahead(config.hidden_size, context=config.lookahead_context),
            torch.nn.Hardtanh(config.tanh_min, config.tanh_max, inplace=True)
        )

        # Layer 10: Fully connected layer
        self.fc = SequenceWise(torch.nn.Sequential(
            torch.nn.BatchNorm1d(config.hidden_size),
            torch.nn.Linear(config.hidden_size, len(string_processor.chars), bias=False)
        ))

        # Loss function
        self.criterion = torch.nn.CTCLoss(blank=string_processor.blank_id, reduction='sum', zero_infinity=True)

        # Decoder used for validation
        self.evaluation_decoder = GreedyDecoder(string_processor.chars, blank_index=string_processor.blank_id)
        self.wer = WordErrorRate(decoder=self.evaluation_decoder, target_decoder=self.evaluation_decoder)
        self.cer = CharErrorRate(decoder=self.evaluation_decoder, target_decoder=self.evaluation_decoder)

        # Datasets
        self.train_dataset = LibriSpeechDataset(
            string_processor, config, train=True,
            filepath='/home/thijs/Datasets/LibriSpeech/train_transcriptions.tsv'
        )
        self.test_dataset = LibriSpeechDataset(
            string_processor, config,
            filepath='/home/thijs/Datasets/LibriSpeech/test_transcriptions.tsv'
        )
        self.val_dataset = LibriSpeechDataset(
            string_processor, config,
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
            x = x.unsqueeze(0)  # batch, feature, time

        x = x.unsqueeze(1)  # batch, channel, feature, time
        x, n_timesteps = self.conv(x, n_timesteps)  # n_timesteps changes during the convolutions

        batch_size, channels, features, timesteps = x.size()
        x = x.view(batch_size, channels * features, timesteps)  # collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # time, batch, feature

        # If we are executing on one example and the model got an LSTM stating state, this function returns the state
        # right before the lookahead overlaps with the end of the number of timesteps such that the model can be used
        # in a streaming fashion.
        lstm_hn = lstm_cn = None
        if n_timesteps is None and lstm_h0 is not None and lstm_c0 is not None:
            if len(spectrograms.shape) == 2:
                lstm_h0 = lstm_h0.unsqueeze(1)
                lstm_c0 = lstm_c0.unsqueeze(1)

            timestep_for_lstm_output = x.size()[0] - self.config.lookahead_context
            if timestep_for_lstm_output <= 0:
                raise Exception("Could not return LSTM hidden state because frame is too small")

            for i, lstm in enumerate(self.rnns):
                h0, c0 = lstm_h0[i, :, :].unsqueeze(0), lstm_c0[i, :, :].unsqueeze(0)

                # We perform the LSTM operation in two parts, first upto timestep_for_lstm_output and then the rest,
                # because we want to capture the hidden state right after timestep_for_lstm_output for streaming.
                xa, hn, cn = lstm(x[:timestep_for_lstm_output, :, :], h0, c0)  # time, batch, feature
                xb, _, _ = lstm(x[timestep_for_lstm_output:,:,:], hn, cn)  # time, batch, feature
                x = torch.cat((xa, xb))

                lstm_hn = hn if lstm_hn is None else torch.cat((lstm_hn, hn))
                lstm_cn = cn if lstm_cn is None else torch.cat((lstm_cn, cn))

            if len(spectrograms.shape) == 2:
                lstm_hn = lstm_hn.squeeze(1)
                lstm_cn = lstm_cn.squeeze(1)
        else:
            for i, lstm in enumerate(self.rnns):
                x, _, _ = lstm(x, n_timesteps=n_timesteps)  # time, batch, feature

        x = self.lookahead(x)  # time, batch, feature
        x = self.fc(x)  # time, batch, label

        x = x.transpose(0, 1)  # batch, time, label
        x = x.log_softmax(-1)  # batch, time, label

        # Return single value inputs
        if len(spectrograms.shape) == 2:
            x = x.squeeze(0)

        # When n_timesteps is not passed, we are not using padding and masking
        if n_timesteps is not None:
            return x, n_timesteps
        elif lstm_hn is not None and lstm_cn is not None:
            return x, lstm_hn, lstm_cn
        else:
            return x

    def training_step(self, batch, batch_idx):
        spectrograms, labels, n_timesteps, n_labels = batch
        y, n_y = self.forward(spectrograms, n_timesteps=n_timesteps)
        loss = self.criterion(y.transpose(0, 1), labels, n_y, n_labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        spectrograms, labels, n_timesteps, n_labels = batch
        y, n_y = self.forward(spectrograms, n_timesteps=n_timesteps)

        loss = self.criterion(y.transpose(0, 1), labels, n_y, n_labels)
        self.wer(preds=y, preds_sizes=n_y, targets=labels, target_sizes=n_labels)
        self.cer(preds=y, preds_sizes=n_y, targets=labels, target_sizes=n_labels)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_wer', self.wer.compute(), prog_bar=True, on_epoch=True, logger=True)
        self.log('val_cer', self.cer.compute(), prog_bar=True, on_epoch=True, logger=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_dataset,
            num_workers=self.config.num_workers
        )

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
