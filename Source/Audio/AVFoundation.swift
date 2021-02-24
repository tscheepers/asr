import AVFoundation

extension AVAudioPCMBuffer {
    func unsafeToVector() -> [Float] {
        return Array(
            UnsafeBufferPointer(start: self.floatChannelData![0], count:Int(self.frameLength))
        )
    }
}
