import Foundation
import AVFoundation
import CoreML

/// Accoustic model ready for streaming inference
class ASR {

    let model: ASRModel = ASRModel()
    let labels = [" ", "'", "a", "b", "c", "d", "e",
                  "f", "g", "h", "i", "j", "k", "l",
                  "m", "n", "o", "p", "q", "r", "s",
                  "t", "u", "v", "w", "x", "y", "z", "."]

    /// Lookahead context for the lookahead layer at the end of the model
    let lookaheadContext: Int = 20

    /// Times two because of the strides in layer 1
    var lookaheadOverflow: Int {
        return lookaheadContext * timestepStrideFirstLayer
    }

    /// The padding we should add for the first convolutional layer.
    /// This padding should be equal to the last frames of the previous chunck, and the first frames of the next chunck
    let padding: Int = 15

    /// Timestep stride in the first layer
    let timestepStrideFirstLayer: Int = 2

    /// The number of timesteps we will get output for (minus the padding and the lookahead context)
    let usefulChunckWidth: Int = 60

    /// divide by two because of the strides in the first layer
    var usefulOutputWidth: Int {
        return usefulChunckWidth / timestepStrideFirstLayer
    }

    /// The number of timesteps in a single input chunck
    var totalChunckWidth: Int {
        return padding + usefulChunckWidth + lookaheadOverflow + padding // add left and right padding
    }

    /// Frame window size for generating the spectrogram
    let nFFT: Int = 320

    /// Hop length for generating the spectrogram
    var hopLength: Int {
        return nFFT / 2
    }

    /// Number of input features per timestep (i.e. frame)
    var numInputFeatures: Int {
        return nFFT / 2 + 1
    }

    /// Number of RNN layers defined by the model
    let numRNNLayers: Int = 5

    /// Hidden size of the RNN layers
    let hiddenSize: Int = 1024

    /// Decode speech to text
    func speechToText(wave: [Float]) -> String {
        let spectrogram = convertToSpectrogram(wave: wave)
        let probabilities = inference(spectrogram: spectrogram)
        return greedyDecode(probabilities: probabilities)
    }

    /// Convert wave to spectrogram
    func convertToSpectrogram(wave: [Float]) -> Matrix<Float>
    {
        let magnitude = wave.shortTimeFourierTransform(nFFT: nFFT, hopLength: hopLength, window: .hamming).magnitude()
        let sepectrogram = log(1 + magnitude)
        let normalized = (sepectrogram - sepectrogram.mean) / sepectrogram.std
        return normalized
    }

    /// Call the model using CoreML
    func inference(spectrogram: Matrix<Float>) -> Matrix<Float> {

        let transposed = spectrogram.width == numInputFeatures && spectrogram.height != numInputFeatures
        assert((transposed && spectrogram.width == numInputFeatures) || (!transposed && spectrogram.height == numInputFeatures))

        let totalFrames = transposed ? spectrogram.height : spectrogram.width
        let iterations = (totalFrames - lookaheadOverflow) / usefulChunckWidth + 1

        var hn = Matrix<Float>.zeros(shape: (numRNNLayers, hiddenSize)).createMLMultiArray()
        var cn = Matrix<Float>.zeros(shape: (numRNNLayers, hiddenSize)).createMLMultiArray()
        var flat: [Float] = []

        // Example below illustrates how to get the correct result from 3 chuncks:
        // -----|--------------------------------|-----
        //   0  | Entire input spectrogram       |  0
        // -----|--------|-----------------------|-----
        // | LP | UI     | LO  | RP |
        // -----|--------|--------|--------------------
        //          | LP | UI     | LO  | RP |
        // --------------|--------|--------------|-----
        //                   | LP | UI     | LO  | RP |
        // -----------------------|--------------|-----
        // PL = Left padding, RP = Right padding
        // UI = Useful input, LO = Lookahead overflow
        //
        // Outputs from iterations:
        // -----|--------|--------|--------------|-----
        //      | 1      | 2      | 3            |
        // -----|--------|--------|--------------|-----
        for i in 0..<iterations {
            let startIdx = (i == 0 ? 0 : i * usefulChunckWidth - padding)
            let endIdx = (i == iterations - 1 ? totalFrames : (i + 1) * usefulChunckWidth + lookaheadOverflow + padding)

            let iterStartIdx = (i == 0 ? padding : 0)
            let iterEndIdx = iterStartIdx + endIdx - startIdx

            var iterSpectrogram = Matrix<Float>.zeros(shape: (numInputFeatures, totalChunckWidth))

            // Fill iter spectrogram
            for row in 0..<numInputFeatures {
                for col in iterStartIdx..<iterEndIdx {
                    let spectrogramCol = startIdx + col - iterStartIdx

                    // The input spectrogram is transposed
                    iterSpectrogram[row,col] = transposed ? spectrogram[spectrogramCol, row] : spectrogram[row, spectrogramCol]
                }
            }

            // Pass in lstm_hn and lstm_cn from the previous iteration, so the hidden states are preserved
            let output = try! model.predictions(inputs: [
                ASRModelInput(spectrogram: iterSpectrogram.createMLMultiArray(), h0: hn, c0: cn)
            ])

            hn = output[0]._712
            cn = output[0]._714

            // Omit the output with lookahead overflow
            let until = (i == iterations - 1 ? (endIdx - startIdx - (i == 0 ? 0 : 1) * padding) : usefulOutputWidth)

            let outputMatrix = output[0]._765.toFloatMatrix()

            // Concatenate the final result
            flat += Array(outputMatrix.flat.prefix(until * labels.count))

        }

        return Matrix<Float>(shape: (flat.count/labels.count, labels.count), flat: flat)
    }

    /// Call the model using CoreML
    func inferenceChunck(
        spectrogram input: Matrix<Float>,
        hn: Matrix<Float>,
        cn: Matrix<Float>
    ) -> (output: Matrix<Float>, hn: Matrix<Float>, cn: Matrix<Float>) {

        let transposed = input.width == numInputFeatures && input.height != numInputFeatures
        let spectrogram = transposed ? input.transposed() : input

        assert(spectrogram.height == numInputFeatures && spectrogram.width == totalChunckWidth)
        assert(hn.height == numRNNLayers && hn.width == hiddenSize)
        assert(cn.height == numRNNLayers && cn.width == hiddenSize)

        // Pass in lstm_hn and lstm_cn from the previous iteration, so the hidden states are preserved
        let output = try! model.predictions(inputs: [
            ASRModelInput(spectrogram: spectrogram.createMLMultiArray(), h0: hn.createMLMultiArray(), c0: cn.createMLMultiArray())
        ])

        // Omit the output with lookahead overflow
        let flat = Array(output[0]._765.toFloatMatrix().flat.prefix(usefulOutputWidth * labels.count))

        return (
            output: Matrix<Float>(shape: (flat.count/labels.count, labels.count), flat: flat),
            hn: output[0]._712.toFloatMatrix(),
            cn: output[0]._714.toFloatMatrix()
        )
    }

    /// Decode from probabilities per time frame
    func greedyDecode(probabilities: Matrix<Float>) -> String {

        var prevChar: Int? = nil
        var decoded: [Int] = []
        let blankIdx = labels.firstIndex(of: ".")

        // Connectionist temporal classification (CTC) collapse
        for char in probabilities.argmax() {
            if char != prevChar && char != blankIdx {
                decoded.append(char)
            }
            prevChar = char
        }

        return decoded.map({ labels[$0] }).joined()
    }
}
