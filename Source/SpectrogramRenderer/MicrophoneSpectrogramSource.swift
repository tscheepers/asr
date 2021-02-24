import Foundation
import AVFoundation

class MicrophoneSpectrogramSource {

// TO-DO:
//    func texture(forPresentationBy renderer: SpectrogramRenderer) -> MTLTexture {
//
//    }

    let session: AVAudioSession = AVAudioSession.sharedInstance()
    let audioEngine = AVAudioEngine()

    /// Start creating a rolling spectrogram
    func start() {
        try! session.setCategory(.playAndRecord, mode: .default)
        try! session.setActive(true)
        // Make sure the sample rate is equal to the sample rate our model expects
        try! session.setPreferredSampleRate(16_000)
        
        switch session.recordPermission {
            case .granted:
                startListening()
            case .denied:
                fatalError("No permission to access the microphone")
            case .undetermined:
                askForPermissionAndStart()
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
        // just in case, assert that the sample rate is 16000
        assert(inputFormat.sampleRate == 16_000)

        inputNode.installTap(onBus: bus, bufferSize: 160, format: inputFormat) { (buffer: AVAudioPCMBuffer, time: AVAudioTime) in
            print(buffer.unsafeToVector())
        }

        audioEngine.prepare()
        try! audioEngine.start()
    }
}

