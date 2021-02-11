import Foundation
import AVFoundation

class Fixtures {

    /// An audiofile from the LibriSpeech dataset
    static let libriSpeechSample : [Float] = Fixtures.loadAudioWave(named: "librispeech-sample")

    /// Generated output to test STFT (generated with librosa)
    static let stftTestOutput : [[Complex<Double>]] = Fixtures.loadComplexCSV(named: "STFTOutput")

    /// Method to load data from CSV
    static func loadCSV(named: String) -> [[Float]] {

        let url =  Bundle(for: Self.self).url(forResource: named, withExtension: "csv")
        let contents = try! String(contentsOf: url!)

        return contents
            .components(separatedBy: "\n")
            .compactMap { $0 == "" ? nil : $0.components(separatedBy: ",").map { Float(Double($0.trimmingCharacters(in: .whitespaces))!) } }
    }

    /// Method to load complex data from CSV
    static func loadComplexCSV(named: String) -> [[Complex<Double>]] {
        let url =  Bundle(for: Self.self).url(forResource: named, withExtension: "csv")
        let contents = try! String(contentsOf: url!)

        return contents
            .components(separatedBy: "\n")
            .compactMap { $0 == "" ? nil : $0.components(separatedBy: ",").map { s -> Complex<Double> in
                let c = s.components(separatedBy: ";").map { $0.trimmingCharacters(in: .whitespaces) }
                return (Double(c.first!)!, Double(c.last!)!)
            } }
    }

    /// Method to load audio wave from wav file
    static func loadAudioWave(named: String, sampleRate: Double? = 16_000) -> [Float] {
        let url =  Bundle(for: Self.self).url(forResource: named, withExtension: "wav")
        let file = try! AVAudioFile(forReading: url!)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate ?? file.fileFormat.sampleRate, channels: 1, interleaved: false)!

        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1024 * 1024)!
        try! file.read(into: buf)

        return Array(UnsafeBufferPointer(start: buf.floatChannelData![0], count:Int(buf.frameLength)))
    }

}

