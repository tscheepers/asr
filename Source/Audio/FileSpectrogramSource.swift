import Foundation
import AVFoundation
import Metal

class FileSpectrogramSource: SpectrogramRendererDelegate {

    private let texture: MTLTexture

    init(named: String, device: MTLDevice) {
        let wave = Self.loadAudioWave(named: named, sampleRate: 16_000)
        let spectrogram: Matrix<Float> = wave.shortTimeFourierTransform(nFFT: 320, hopLength: 160, window: .hamming).magnitude()

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.width = spectrogram.shape.width
        textureDescriptor.height = spectrogram.shape.height
        textureDescriptor.depth = 1

        texture = device.makeTexture(descriptor: textureDescriptor)!
        spectrogram.fill(texture: texture)
    }

    func texture(forPresentationBy renderer: SpectrogramRenderer) -> MTLTexture {
        return texture
    }

    /// Method to load audio wave from wav file
    static func loadAudioWave(named: String, sampleRate: Double? = 16_000, fileExtension: String = "wav") -> [Float] {
        let url =  Bundle(for: Self.self).url(forResource: named, withExtension: fileExtension)
        let file = try! AVAudioFile(forReading: url!)

        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate ?? file.fileFormat.sampleRate, channels: 1, interleaved: false)!
        let capacity: UInt32 = sampleRate != nil ? UInt32(Double(file.length) * sampleRate! / file.fileFormat.sampleRate) : UInt32(file.length)

        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity)!
        try! file.read(into: buf)

        return buf.unsafeToVector()
    }
}