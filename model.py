import pytorch_lightning
import torch
from common_voice_dataset import CommonVoiceDataset, pad_dataset
from qualitative_evaluation import QualitativeEvaluation


class Model(pytorch_lightning.core.lightning.LightningModule):

    def __init__(self, batch_size=64, hidden_size=256, num_classes=28, n_features=64,
                 num_layers=2, dropout=0.1, sample_rate=16000, max_timesteps=3000):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.max_timesteps = max_timesteps
        self.n_features = n_features

        self.rnn = torch.nn.GRU(input_size=n_features, hidden_size=hidden_size,
                                num_layers=num_layers, dropout=dropout)

        self.final_fc = torch.nn.Linear(hidden_size, num_classes)

        self.criterion = torch.nn.CTCLoss(blank=27, zero_infinity=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, hidden):
        x = x.transpose(0, 2).transpose(1, 2) # time, batch, feature

        z, hidden = self.rnn(x, hidden)

        y = self.final_fc(z)

        log_probabilities = torch.nn.functional.log_softmax(y, dim=2)

        return log_probabilities, hidden

    def step(self, batch):
        features, labels, n_features, n_labels = batch
        batch_size = features.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        log_probabilities, _ = self.forward(features, h0)

        loss = self.criterion(log_probabilities, labels, n_features, n_labels)

        return loss, log_probabilities

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch)
        return {'loss': loss}

    def train_dataloader(self):
        dataset = CommonVoiceDataset(
            filename='train.tsv',
            n_features=self.n_features,
            sample_rate=self.sample_rate,
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
        dataset = CommonVoiceDataset(
            filename='dev.tsv',
            n_features=self.n_features,
            sample_rate=self.sample_rate,
            max_timesteps=self.max_timesteps
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=pad_dataset,
            num_workers=16
        )

    def test_dataloader(self):
        dataset = CommonVoiceDataset(
            filename='test.tsv',
            n_features=self.n_features,
            sample_rate=self.sample_rate,
            max_timesteps=self.max_timesteps
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=pad_dataset,
            num_workers=16
        )


if __name__ == '__main__':
    model = Model()
    pytorch_lightning.Trainer(
        max_epochs=1000, gpus=1,
        num_nodes=1, distributed_backend=None,
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=5,
        callbacks=[QualitativeEvaluation()]
    ).fit(model)
