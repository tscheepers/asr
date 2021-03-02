import UIKit
import Metal
import AVFoundation
import Combine

class SpectrogramViewController: UIViewController {

    let renderer: SpectrogramRenderer = SpectrogramRenderer()

    var spectrogramSource: SpectrogramRendererDataSource!

    var metalLayer: CAMetalLayer!

    var timer: CADisplayLink?

    var zoomRecognizer: UIPinchGestureRecognizer!

    var model: ASR!

    /// Combine cancellables
    var cancellables = Set<AnyCancellable>()

    /// Buffer for holding the most recently received spectrogram so it cen be fed into the model
    var spectrogramBuffer: [Float] = []
    var hn: Matrix<Float>!
    var cn: Matrix<Float>!

    override func viewDidLoad() {
        super.viewDidLoad()

        model = ASR()

        zoomRecognizer = UIPinchGestureRecognizer(target: self, action: #selector(SpectrogramViewController.zoom(gestureRecognizer:)))
        view.addGestureRecognizer(zoomRecognizer)

        let microphoneSpectrogramSource = MicrophoneSpectrogramSource(device: renderer.device)
        // spectrogramSource = FileSpectrogramSource(named: "librispeech-sample", device: renderer.device)
        spectrogramSource = microphoneSpectrogramSource
        renderer.dataSource = spectrogramSource

        startListening(source: microphoneSpectrogramSource)

        metalLayer = renderer.createMetalLayer(frame: view.layer.frame)
        view.layer.addSublayer(metalLayer)
        view.clipsToBounds = true

        timer = CADisplayLink(target: self, selector: #selector(gameloop))
        timer?.add(to: RunLoop.main, forMode: .default)
    }

    func startListening(source: MicrophoneSpectrogramSource) {
        // Pass the microphone data to the model
        let shape = (self.model.numRNNLayers, self.model.hiddenSize)
        hn = Matrix<Float>.zeros(shape: shape)
        cn = Matrix<Float>.zeros(shape: shape)

        // TO-DO: Move this elsewhere
        source.spectrogramSubject.receive(on: DispatchQueue.global()).sink { (error) in
            fatalError("Cannot receive anymore microphone data")
        } receiveValue: { [weak self] (spectrogram: Matrix<Float>) in
            guard let self = self else { return }
            assert(spectrogram.width == self.model.numInputFeatures)
            self.spectrogramBuffer += spectrogram.flat

            while self.spectrogramBuffer.count >= self.model.numInputFeatures * self.model.totalChunckWidth {
                let input = Array(self.spectrogramBuffer.prefix(self.model.numInputFeatures * self.model.totalChunckWidth))
                let shape = (self.model.totalChunckWidth, self.model.numInputFeatures)
                let result = self.model.inferenceChunck(spectrogram: Matrix(shape: shape, flat: input), h0: self.hn, c0: self.cn)

                self.hn = result.hn
                self.cn = result.cn

                print(result.output.argmax().map({ self.model.labels[$0] }).joined())

                let keep = self.spectrogramBuffer.count - (self.model.usefulChunckWidth + self.model.padding) * self.model.numInputFeatures
                self.spectrogramBuffer = Array(self.spectrogramBuffer.suffix(keep))
            }
        }.store(in: &cancellables)

        // Start listening
        source.start()
    }

    override func viewDidDisappear(_ animated: Bool) {
        timer?.remove(from: RunLoop.main, forMode: .default)
        timer = nil
    }

    private let minZoom: Double = 0.05
    private let maxZoom: Double = 1.0
    private var zoomStart: Double = 1.0
    private var zoom: Double = 1.0 {
        didSet {
            renderer.zoom = Float(zoom)
        }
    }

    /// Allow pinching to zoom the spectrogram
    @objc func zoom(gestureRecognizer: UIPinchGestureRecognizer) {
        // We make sure there is single value describing the zoom state, across different pinch gestures.
        switch gestureRecognizer.state {
        case .began,
             .changed:
            zoom = fmin(fmax(zoomStart * 1.0 / Double(gestureRecognizer.scale), minZoom), maxZoom)
        case .ended:
            zoomStart = zoom
        case .cancelled,
             .failed:
            zoom = zoomStart
        default:
            return
        }
    }

    @objc func gameloop() {
      autoreleasepool {
        guard let drawable = metalLayer?.nextDrawable() else { return }
        renderer.render(drawable: drawable)
      }
    }

}
