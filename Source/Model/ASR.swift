import Foundation
import AVFoundation
import CoreML

class ASR {

    let model: ASRModel = ASRModel()
    let labels = ["_", "'", "A", "B", "C", "D", "E",
                  "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S",
                  "T", "U", "V", "W", "X", "Y", "Z", " "]

    /// Decode speech to text
    func speechToText(wave: [Float]) -> String {
        let spectrogram = convertToSpectrogram(wave: wave)
        let probabilities = inference(spectrogram: spectrogram)
        return greedyDecode(probabilities: probabilities)
    }

    /// Convert wave to spectrogram
    func convertToSpectrogram(wave: [Float]) -> Matrix<Float>
    {
        let magnitude = wave.shortTimeFourierTransform(nFFT: 320, hopLength: 160, window: .hamming).magnitude()
        let sepectrogram = log(1 + magnitude)
        let normalized = (sepectrogram - sepectrogram.mean) / sepectrogram.std
        return normalized
    }

    /// Call the model using CoreML
    func inference(spectrogram: Matrix<Float>) -> Matrix<Float> {

        let frames = 50
        let steps = spectrogram.height / frames
        var hn = Matrix<Float>.zeros(shape: (5, 1024))
        var cn = Matrix<Float>.zeros(shape: (5, 1024))
        var flat: [Float] = []

        for i in 0..<steps {
            let s = Matrix<Float>(shape: (frames, 161), flat: Array(spectrogram.flat[i*161*frames..<(i+1)*161*frames])).transposed()

            let output = try! model.predictions(inputs: [ASRModelInput(spectrogram: s.createMLMultiArray(), h0: hn.createMLMultiArray(), c0: cn.createMLMultiArray())])

            let outputMatrix = output[0]._385.toFloatMatrix()
            flat += outputMatrix.flat
            hn = output[0]._387.toFloatMatrix()
            cn = output[0]._389.toFloatMatrix()
        }

        return Matrix<Float>(shape: (steps*(frames/2), 29), flat: flat)
    }

    /// Decode from probabilities per time frame
    func greedyDecode(probabilities: Matrix<Float>) -> String {

        var prevChar: Int? = nil
        var decoded: [Int] = []

        // Connectionist temporal classification (CTC) collapse
        for char in probabilities.argmax() {
            if char != prevChar && char != 0 {
                decoded.append(char)
            }
            prevChar = char
        }

        return decoded.map({ labels[$0] }).joined()
    }
}
