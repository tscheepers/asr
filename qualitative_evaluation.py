import pytorch_lightning
import torch
from ctcdecode import CTCBeamDecoder
from dataset import LibriSpeechDataset
import Levenshtein as Lev


class QualitativeEvaluation(pytorch_lightning.callbacks.Callback):

    def __init__(self, n_samples=2):
        self.tiny_dataset = None
        self.string_processor = None
        self.beam_search = None
        self.n_samples = n_samples

    def setup(self, trainer, model, stage: str):
        self.string_processor = model.string_processor

        self.tiny_dataset = LibriSpeechDataset(
            model.string_processor,
            n_features=model.n_features,
            sample_rate=model.sample_rate,
            max_timesteps=model.max_timesteps,
            sample_limit=self.n_samples
        )

        self.beam_search = CTCBeamDecoder(self.string_processor.chars, beam_width=100,
                                          blank_id=self.string_processor.blank_id, log_probs_input=True)

    def on_validation_epoch_end(self, trainer, model):
        with torch.no_grad():
            for (features, labels, n_features, _) in self.tiny_dataset:

                features = features.unsqueeze(0).to(model.device)
                y, _ = model(features, torch.IntTensor([n_features]))
                log_probabilities = torch.nn.functional.log_softmax(y, dim=2)

                ground_truth = self.string_processor.labels_to_str(labels)
                print("orig.:\t\t\"%s\"" % ground_truth)

                arg_maxes = torch.argmax(log_probabilities.squeeze(1), 1).cpu().numpy()
                output = self.string_processor.labels_to_str(arg_maxes)
                greedy = self.string_processor.labels_to_str(collapse_ctc(arg_maxes, blank_id=self.string_processor.blank_id))
                print("output:\t\t\"%s\"" % output)
                print("greedy:\t\t\"%s\" cer: %f wer: %f" % (greedy, calc_cer(greedy, ground_truth), calc_wer(greedy, ground_truth)))

                beam_result, beam_scores, timesteps, out_seq_len = self.beam_search.decode(log_probabilities.transpose(0, 1))
                for i in range(5):
                    result = beam_result[0][i][0:out_seq_len[0][i]].numpy().tolist()
                    string = self.string_processor.labels_to_str(result)
                    print("beam %d:\t\t\"%s\"  cer: %f wer: %f" % (i, string, calc_cer(greedy, ground_truth), calc_wer(greedy, ground_truth)))

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


def calc_wer(hyp, ref):
    # build mapping of words to integers
    b = set(hyp.split() + ref.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts strings)
    w1 = [chr(word2char[w]) for w in hyp.split()]
    w2 = [chr(word2char[w]) for w in ref.split()]

    distance = Lev.distance(''.join(w1), ''.join(w2))
    return distance / len(ref.split()) * 100


def calc_cer(hyp, ref):
    hyp, ref, = hyp.replace(' ', ''), ref.replace(' ', '')
    distance = Lev.distance(hyp, ref)
    return distance / len(ref) * 100

