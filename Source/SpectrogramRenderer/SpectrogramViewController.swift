import UIKit
import Metal
import AVFoundation

class SpectrogramViewController: UIViewController {

    var renderer: SpectrogramRenderer = SpectrogramRenderer()

    var spectrogramSource: MicrophoneSpectrogramSource = MicrophoneSpectrogramSource()

    var metalLayer: CAMetalLayer!

    var timer: CADisplayLink?

    override func viewDidLoad() {
        super.viewDidLoad()

        // TO-DO: renderer.delegate = spectrogramSource
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

    @objc func gameloop() {
      autoreleasepool {
        guard let drawable = metalLayer?.nextDrawable() else { return }
        renderer.render(drawable: drawable)
      }
    }

}
