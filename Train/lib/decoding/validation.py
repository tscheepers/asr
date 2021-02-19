#!/usr/bin/env python
# ----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Sean Naren
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

import pytorch_lightning
import torch
import Levenshtein as Lev
from abc import ABC, abstractmethod
from torch.cuda.amp import autocast
from tqdm import tqdm
from decoder import Decoder, GreedyDecoder


class ErrorRate(pytorch_lightning.metrics.Metric, ABC):
    def __init__(self,
                 decoder: Decoder,
                 target_decoder: GreedyDecoder,
                 save_output: bool = False,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output

    @abstractmethod
    def calculate_metric(self, transcript, reference):
        raise NotImplementedError

    def update(self, preds: torch.Tensor,
               preds_sizes: torch.Tensor,
               targets: torch.Tensor,
               target_sizes: torch.Tensor):
        decoded_output, _ = self.decoder.decode(preds, preds_sizes)
        target_strings = self.target_decoder.convert_to_strings(targets)
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            self.calculate_metric(
                transcript=transcript,
                reference=reference
            )


class CharErrorRate(ErrorRate):
    def __init__(self,
                 decoder: Decoder,
                 target_decoder: GreedyDecoder,
                 save_output: bool = False,
                 dist_sync_on_step: bool = False):
        super().__init__(
            decoder=decoder,
            target_decoder=target_decoder,
            save_output=save_output,
            dist_sync_on_step=dist_sync_on_step
        )
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output
        self.add_state("cer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_chars", default=torch.tensor(0), dist_reduce_fx="sum")

    def calculate_metric(self, transcript, reference):
        cer_inst = self.cer_calc(transcript, reference)
        self.cer += cer_inst
        self.n_chars += len(reference.replace(' ', ''))

    def compute(self):
        cer = float(self.cer) / self.n_chars
        return cer.item() * 100

    def cer_calc(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)


class WordErrorRate(ErrorRate):
    def __init__(self,
                 decoder: Decoder,
                 target_decoder: GreedyDecoder,
                 save_output: bool = False,
                 dist_sync_on_step: bool = False):
        super().__init__(
            decoder=decoder,
            target_decoder=target_decoder,
            save_output=save_output,
            dist_sync_on_step=dist_sync_on_step
        )
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output
        self.add_state("wer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def calculate_metric(self, transcript, reference):
        wer_inst = self.wer_calc(transcript, reference)
        self.wer += wer_inst
        self.n_tokens += len(reference.split())

    def compute(self):
        wer = float(self.wer) / self.n_tokens
        return wer.item() * 100

    def wer_calc(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))


@torch.no_grad()
def run_evaluation(test_loader,
                   model,
                   decoder: Decoder,
                   device: torch.device,
                   target_decoder: Decoder,
                   precision: int):
    model.eval()
    wer = WordErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    cer = CharErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        with autocast(enabled=precision == 16):
            out, output_sizes = model(inputs, input_sizes)
        decoded_output, _ = decoder.decode(out, output_sizes)
        wer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        cer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
    return wer.compute(), cer.compute()
