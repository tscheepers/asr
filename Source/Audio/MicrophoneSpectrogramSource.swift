import Foundation
import AVFoundation
import Accelerate
import Metal

class MicrophoneSpectrogramSource {

// TO-DO:
//    func texture(forPresentationBy renderer: SpectrogramRenderer) -> MTLTexture {
//
//    }

    let frameLength: Int
    let sampleRate: Double

    /// All the data received thusfar.
    private var spectrogramColumns: [[Float]] = []

    /// Used for processing a new frame. Each frame consists of two parts, the previous wave and the new wave.
    /// So we need to keep track of the previous wave we received from the mic.
    private var previousWave: [Float]?

    /// Used for optimizing the FFT algorithm.
    private var fftSetup: vDSP_DFT_Setup?

    /// Used for tampering each frame to reduce edge artifacts.
    private(set) lazy var hammingWindow: [Float] = {
        [Float].hammingWindow(windowLength: self.frameLength)
    }()

    /// Used for checking permissions
    let session: AVAudioSession = AVAudioSession.sharedInstance()

    /// Used for tapping the microphone
    let audioEngine = AVAudioEngine()

    init(frameLength: Int = 320, sampleRate: Double = 16_000) {
        self.frameLength = frameLength
        self.sampleRate = sampleRate
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

    /// Listen to the audio from the microphone
    private func startListening() {
        let inputNode = audioEngine.inputNode // This automatically defaults to the microphone
        let bus: AVAudioNodeBus = 0
        let inputFormat = inputNode.inputFormat(forBus: bus)

        // AVFoundation does not guarantee the sample rate to equal 16000 but it seems to work
        // just in case, assert that this is the case
        assert(inputFormat.sampleRate == self.sampleRate)

        // Should be exactly halve since we are combining two frames continuously
        let frameLength = AVAudioFrameCount(frameLength / 2)

        inputNode.installTap(onBus: bus, bufferSize: frameLength, format: inputFormat) { (buffer: AVAudioPCMBuffer, time: AVAudioTime) in
            let wave = buffer.unsafeToVector()
            if let previousWave = self.previousWave {
                // Combine the previous wave with the current wave and process
                self.process(newFrame: (wave + previousWave))
            }
            self.previousWave = wave
        }

        audioEngine.prepare()
        try! audioEngine.start()
    }

    // Process new frame comming in
    func process(newFrame frame: [Float]) {

        // Assert frame length
        assert(frame.count == self.frameLength)

        // Tamper using the Hamming function
        let inputFrame: [Float] = zip(frame, hammingWindow).map { $0.0 * $0.1 }

        // Execute the fourier transform
        let (reals, imags, fftSetup) = [Float].discreteFourierTransform(reals: inputFrame, reuseSetup: self.fftSetup)
        self.fftSetup = fftSetup // For reuse next time

        // Generate a spectrogram column
        let magnitudes: [Float] = zip(reals, imags).map { sqrt($0.0 * $0.0 + $0.1 * $0.1) }

        // Add a new column of the spectrogram
        self.spectrogramColumns.append(magnitudes)
    }
}


