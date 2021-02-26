import UIKit
import Metal
import AVFoundation

class SpectrogramViewController: UIViewController {

    let renderer: SpectrogramRenderer = SpectrogramRenderer()

    var spectrogramSource: SpectrogramRendererDataSource!

    var metalLayer: CAMetalLayer!

    var timer: CADisplayLink?

    var zoomRecognizer: UIPinchGestureRecognizer!

    override func viewDidLoad() {
        super.viewDidLoad()

        zoomRecognizer = UIPinchGestureRecognizer(target: self, action: #selector(SpectrogramViewController.zoom(gestureRecognizer:)))
        view.addGestureRecognizer(zoomRecognizer)

        let microphoneSpectrogramSource = MicrophoneSpectrogramSource(device: renderer.device)
        // spectrogramSource = FileSpectrogramSource(named: "librispeech-sample", device: renderer.device)

        spectrogramSource = microphoneSpectrogramSource
        renderer.dataSource = spectrogramSource

        // Start listening
        microphoneSpectrogramSource.start()

        metalLayer = renderer.createMetalLayer(frame: view.layer.frame)
        view.layer.addSublayer(metalLayer)
        view.clipsToBounds = true

        timer = CADisplayLink(target: self, selector: #selector(gameloop))
        timer?.add(to: RunLoop.main, forMode: .default)
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
