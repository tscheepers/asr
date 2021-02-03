import pytorch_lightning
import torch
from ctcdecode import CTCBeamDecoder
from common_voice_dataset import CommonVoiceDataset, StringProcessor


class QualitativeEvaluation(pytorch_lightning.callbacks.Callback):

    def __init__(self, n_samples=2):
        self.tiny_dataset = None
        self.n_samples = n_samples
        self.string_processor = StringProcessor()
        self.beam_search = CTCBeamDecoder(self.string_processor.chars, beam_width=100, blank_id=27, log_probs_input=True)

    def setup(self, trainer, model, stage: str):
        self.tiny_dataset = CommonVoiceDataset(
            filename='dev.tsv',
            n_features=model.n_features,
            sample_rate=model.sample_rate,
            max_timesteps=model.max_timesteps,
            sample_limit=self.n_samples
        )

    def on_validation_epoch_end(self, trainer, model):
        with torch.no_grad():
            for (features, labels, n_features, n_labels) in self.tiny_dataset:
                h0 = torch.zeros(model.num_layers, 1, model.hidden_size).to(model.device)
                features = features.unsqueeze(0).to(model.device)
                log_probabilities, _ = model(features, h0)

                ground_truth = self.string_processor.labels_to_str(labels)
                print("orig.:\t\t\"%s\"" % ground_truth)

                arg_maxes = torch.argmax(log_probabilities.squeeze(1), 1).cpu().numpy()
                output = self.string_processor.labels_to_str(arg_maxes)
                greedy = self.string_processor.labels_to_str(collapse_ctc(arg_maxes))
                print("output:\t\t\"%s\"" % output)
                print("greedy:\t\t\"%s\"" % greedy)

                beam_result, beam_scores, timesteps, out_seq_len = self.beam_search.decode(log_probabilities.transpose(0, 1))
                for i in range(5):
                    result = beam_result[0][i][0:out_seq_len[0][i]].numpy().tolist()
                    string = self.string_processor.labels_to_str(result)
                    print("beam %d:\t\t\"%s\"" % (i, string))

                print('')


def collapse_ctc(arg_maxes, blank_id=27):
    result = []
    for i, index in enumerate(arg_maxes):
        if index == blank_id:
            continue
        if i != 0 and index == arg_maxes[i - 1]:
            continue
        result.append(index.item())
    return result