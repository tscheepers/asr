#!/usr/bin/python

import sys
import torch
import coremltools as ct
from cnn_rnn_lookahead_acoustic_model import CnnRnnLookaheadAcousticModel as Model
from cnn_rnn_lookahead_acoustic_model import CnnRnnLookaheadAcousticModelConfig as Config


def main(args):
    """
    Converts a pytoch model to a CoreML model
    """

    if len(args) != 2:
        print('Please enter two arguments: coreml.py <inputfile.ckpt> <outputfile.mlmodel>')
        sys.exit(2)

    input_file, output_file = args

    # Load model
    model = Model.load_from_checkpoint(input_file, model_config=Config(time_padding=False))

    # Set model to evaluation mode
    model.eval()
    model.zero_grad()
    torch.set_grad_enabled(False)

    # On inference the user is expected to pad the input
    useful_frame_width = 60
    padding = 15
    lookahead_overflow = model.model_config.lookahead_context * 2  # times two because of the strides in layer 1
    total_frame_width = padding + useful_frame_width + lookahead_overflow + padding  # add left and right padding

    # Input descriptions
    spectrogram = torch.rand((161, total_frame_width))
    h0 = torch.zeros(model.model_config.num_rnn_layers, model.model_config.hidden_size)
    c0 = torch.zeros(model.model_config.num_rnn_layers, model.model_config.hidden_size)

    # Trace model
    traced_model = torch.jit.trace(model, (spectrogram, h0, c0))

    # Convert model
    coreml_model = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[
            ct.TensorType(name="spectrogram", shape=spectrogram.shape),
            ct.TensorType(name="h0", shape=h0.shape),
            ct.TensorType(name="c0", shape=c0.shape),
        ],
        output_names=["log_probabilities", "hn", "cn"]  # output_names do not seem to work yet
    )
    coreml_model.save(output_file)


if __name__ == '__main__':
    main(sys.argv[1:])
