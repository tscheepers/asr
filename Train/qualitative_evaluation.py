import pytorch_lightning
import torch
from ctcdecode import CTCBeamDecoder
from model.cnn_rnn_lookahead_acoustic_model import CnnRnnLookaheadAcousticModel as Model
import Levenshtein as Lev


class QualitativeEvaluator:
    """
    Use the QualitativeEvaluator for evaluating, processing and printing a couple of validation samples during training.
    This can be helpful to see if the model is producing sensible output.
    """

    def __init__(self, model: Model, beam_search: CTCBeamDecoder = None, print_beams: int = 5):
        self.model = model
        self.string_processor = model.string_processor
        self.beam_search = CTCBeamDecoder(
            model.string_processor.chars,
            beam_width=100,
            blank_id=model.string_processor.blank_id,
            log_probs_input=True
        ) if beam_search is None else beam_search
        self.print_beams = print_beams

    def print_evaluation_of_sample(self, spectrogram, labels, split_every=None):
        """
        Evaluate the model using reference labels and print their result to the console
        """
        spectrogram = torch.Tensor(spectrogram).to(self.model.device)
        y = self.model.forward(spectrogram)
        self.print_evaluation_of_output(y, labels, split_every=split_every)

    def print_evaluation_of_output(self, log_probabilities, labels, split_every=30):
        """
        Evaluate the model's output using reference labels and print their result to the console
        """
        arg_maxes, greedy_decoded, beams_decoded = self.decode_log_probabilities(log_probabilities)

        original_string = self.string_processor.labels_to_str(labels)
        print("orig.:\t\t\"%s\"" % original_string)
        print("output:\t\t\"%s\"" % self.string_processor.labels_to_str(arg_maxes, split_every=split_every))

        greedy_string = self.string_processor.labels_to_str(greedy_decoded)
        print("greedy:\t\t\"%s\" [cer: %.2f] [wer: %.2f]" % (
            greedy_string,
            self.character_error_rate(greedy_string, original_string),
            self.word_error_rate(greedy_string, original_string)
        ))

        for i, beam_decoded in enumerate(beams_decoded):
            string = self.string_processor.labels_to_str(beam_decoded)
            print("beam %d:\t\t\"%s\"  [cer: %.2f] [wer: %.2f]" % (
                (i + 1), string,
                self.character_error_rate(string, original_string),
                self.word_error_rate(string, original_string)
            ))

    def decode_log_probabilities(self, log_probabilities):
        """
        Decodes the model output
        """
        arg_maxes = torch.argmax(log_probabilities, -1).cpu().numpy()
        greedy_decoded = self.collapse_ctc(arg_maxes, blank_id=self.string_processor.blank_id)

        beams_decoded = []
        beam_result, _, _, out_seq_len = self.beam_search.decode(log_probabilities.unsqueeze(0))
        for i in range(self.print_beams):
            beams_decoded.append(
                beam_result[0][i][0:out_seq_len[0][i]].numpy().tolist()
            )

        return arg_maxes, greedy_decoded, beams_decoded

    @staticmethod
    def collapse_ctc(arg_maxes, blank_id=28):
        """
        Collapses a string like "...aaaa.b.c.c.ddd.e.." into "abccde"
        """
        result = []
        for i, index in enumerate(arg_maxes):
            if index == blank_id:
                continue
            if i != 0 and index == arg_maxes[i - 1]:
                continue
            result.append(index.item())
        return result

    @staticmethod
    def word_error_rate(hyp, ref):
        """
        Calculates the word error rate for a particular sentence
        """
        # build mapping of words to integers
        b = set(hyp.split() + ref.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts strings)
        w1 = [chr(word2char[w]) for w in hyp.split()]
        w2 = [chr(word2char[w]) for w in ref.split()]

        distance = Lev.distance(''.join(w1), ''.join(w2))
        return distance / len(ref.split()) * 100

    @staticmethod
    def character_error_rate(hyp, ref):
        """
        Calculates the character error rate for a particular sentence
        """
        hyp, ref, = hyp.replace(' ', ''), ref.replace(' ', '')
        distance = Lev.distance(hyp, ref)
        return distance / len(ref) * 100


class QualitativeEvaluationCallback(pytorch_lightning.callbacks.Callback):
    """
    Callback to be used with PyTorchLighting. Every time a validation epoch finishes, this callback will process a
    couple of validation examples and print them to the console.
    """

    def __init__(self, n_samples=2):
        self.dataset = None
        self.qualitative_evaluator = None
        self.n_samples = n_samples

    def setup(self, trainer, model, stage: str):
        self.dataset = model.val_dataset
        self.qualitative_evaluator = QualitativeEvaluator(
            model,
            model.string_processor,
            CTCBeamDecoder(
                model.string_processor.chars,
                beam_width=100,
                blank_id=model.string_processor.blank_id,
                log_probs_input=True
            )
        )

    def on_validation_epoch_end(self, trainer, model):
        with torch.no_grad():
            for i in range(self.n_samples):
                (spectrogram, labels, _, _) = self.dataset[i]
                self.qualitative_evaluator.print_evaluation_of_sample(spectrogram, labels)
