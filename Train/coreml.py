from model import Model
import torch
import coremltools as ct


if __name__ == '__main__':
    model = Model.load_from_checkpoint(
        './checkpoints-long-run-librispeech-without-spec-augment/epoch=11-step=208357.ckpt')

    frames = 500
    spectrogram = torch.rand((161, frames))
    h0 = torch.zeros(model.config.num_layers, model.config.hidden_size)
    c0 = torch.zeros(model.config.num_layers, model.config.hidden_size)
    traced_model = torch.jit.trace(model, (spectrogram, h0, c0))

    cmlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[
            ct.TensorType(name="spectrogram", shape=spectrogram.shape),
            ct.TensorType(name="h0", shape=h0.shape),
            ct.TensorType(name="c0", shape=c0.shape),
        ],
        output_names=["probabilities", "hn", "cn"]
    )

    cmlmodel.author = "Thijs Scheepers"
    cmlmodel.license = "MIT License"

    cmlmodel.save("./coreml/ASRModel.mlmodel")