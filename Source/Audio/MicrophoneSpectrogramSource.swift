import Foundation
import UIKit
import AVFoundation
import Accelerate
import Metal

class MicrophoneSpectrogramSource: SpectrogramRendererDataSource {

    let frameLength: Int
    let sampleRate: Double

    /// Used for checking permissions
    let session: AVAudioSession = AVAudioSession.sharedInstance()

    /// Used for tapping the microphone
    let audioEngine = AVAudioEngine()

    /// We will display as much timesteps (i.e. frames) as there are pixels on the screen
    let textureHeight = Int(UIScreen.main.nativeBounds.width)

    /// Points to the column in the texture we can fill when a new frame arrives
    /// This should always be `0 <= texturePointer < textureWidth`
    var texturePointer: Int = 0

    init(frameLength: Int = 320, sampleRate: Double = 16_000, device: MTLDevice) {
        self.frameLength = frameLength
        self.sampleRate = sampleRate

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.width = (frameLength / 2) + 1 // Since we use the real valued FFT we dismiss half of the values
        textureDescriptor.height = textureHeight
        textureDescriptor.depth = 1

        texture = device.makeTexture(descriptor: textureDescriptor)!
    }

    /// Start creating a rolling spectrogram
    func start() {
        try! session.setCategory(.playAndRecord, mode: .default)
        try! session.setActive(true)
        // Make sure the sample rate is equal to the sample rate our model expects
        try! session.setPreferredSampleRate(self.sampleRate)
        
        switch session.recordPermission {
            case .granted:
                startListening()
            case .denied:
                fatalError("No permission to access the microphone")
            case .undetermined:
                askForPermissionAndStart()
        @unknown default:
            fatalError("Unexpected record permission state")
        }
    }

    /// Presents the permission UI
    private func askForPermissionAndStart() {
        session.requestRecordPermission() { allowed in
            if !allowed {
                fatalError("No permission to access the microphone")
            }
            self.startListening()
        }
    }

    /// Used for processing a new wave to account for the lack of reflection on the STFT
    private var previousWave: [Float]?

    /// Listen to the audio from the microphone
    private func startListening() {
        let inputNode = audioEngine.inputNode // This automatically defaults to the microphone
        let bus: AVAudioNodeBus = 0
        let inputFormat = inputNode.inputFormat(forBus: bus)

        // AVFoundation does not guarantee the sample rate to equal 16000 but it seems to work
        // just in case, assert that this is the case
        assert(inputFormat.sampleRate == self.sampleRate)

        // But installTap has a minimum of 100ms of buffer so we need to chunck after we receive a new buffer
        let frameLength = AVAudioFrameCount(self.frameLength / 2)

        inputNode.installTap(onBus: bus, bufferSize: frameLength, format: inputFormat) { (buffer: AVAudioPCMBuffer, time: AVAudioTime) in
            let wave = buffer.unsafeToVector()
            assert(wave.count % self.frameLength == 0)

            // We add two frames from the previous wave to account to account for disabling reflections on the STFT
            if let previousWave = self.previousWave {
                self.process(newWave: (previousWave + wave))
            } else {
                self.process(newWave: Array(wave))
            }
            self.previousWave = Array(wave.suffix(self.frameLength))
        }

        audioEngine.prepare()
        try! audioEngine.start()
    }

    // Process new frame comming in
    func process(newWave wave: [Float]) {

        // Assert frame length is divisable by the frameLength
        assert(wave.count % self.frameLength == 0)

        let frames = wave.shortTimeFourierTransform(nFFT: self.frameLength, hopLength: self.frameLength/2, window: .hamming, reflect: false)

        // Generate a spectrogram column
        let spectrogram = log(1 + frames.magnitude())

        // TODO: Add normalization using running mean and standard deviation

        // Update texture
        spectrogram.fill(texture: texture, offset: (0, texturePointer % texture.height))
        texturePointer = texturePointer + spectrogram.height
    }

    // MARK: - SpectrogramRendererDataSource
    
    /// The texture containing the spectrogram, ready for rendering
    private(set) var texture: MTLTexture

    var textureHeightOffset: Int {
        return texturePointer
    }
}


