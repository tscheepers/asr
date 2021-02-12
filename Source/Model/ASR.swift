import Foundation
import AVFoundation
import CoreML

class ASR {

    let model: DeepSpeech = DeepSpeech()
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
        let magnitude = wave.shortTimeFourierTransform(nFFT: 320, hopLength: 160).magnitude()
        let sepectrogram = log(1 + magnitude)
        let normalized = (sepectrogram - sepectrogram.mean) / sepectrogram.std
        return normalized
    }

    /// Call the model using CoreML
    func inference(spectrogram: Matrix<Float>) -> Matrix<Float> {
        let spectrogramForCoreML = spectrogram.transposed().createMLMultiArray(prependDimensions: 2)
        let output = try! model.predictions(inputs: [DeepSpeechInput(inputs: spectrogramForCoreML)])
        return output[0]._348.toFloatMatrix()
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
