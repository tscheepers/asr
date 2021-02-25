import UIKit
import Metal
import AVFoundation

class SpectrogramViewController: UIViewController {

    let renderer: SpectrogramRenderer = SpectrogramRenderer()

    var spectrogramSource: MicrophoneSpectrogramSource!

    var metalLayer: CAMetalLayer!

    var timer: CADisplayLink?

    var zoomRecognizer: UIPinchGestureRecognizer!

    override func viewDidLoad() {
        super.viewDidLoad()

        zoomRecognizer = UIPinchGestureRecognizer(target: self, action: #selector(SpectrogramViewController.zoom(gestureRecognizer:)))
        view.addGestureRecognizer(zoomRecognizer)

//        spectrogramSource = FileSpectrogramSource(named: "librispeech-sample", device: renderer.device)
        spectrogramSource = MicrophoneSpectrogramSource(device: renderer.device)
        renderer.delegate = spectrogramSource

        spectrogramSource.start()

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

    private var zoomStart: Double = 1.0
    private var zoom: Double = 1.0 {
        didSet {
            renderer.zoom = zoom
        }
    }

    @objc func zoom(gestureRecognizer: UIPinchGestureRecognizer) {
        switch gestureRecognizer.state {
        case .began,
             .changed:
            zoom = fmin(fmax(zoomStart * 1.0 / Double(gestureRecognizer.scale), 0.05), 1.0)
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
