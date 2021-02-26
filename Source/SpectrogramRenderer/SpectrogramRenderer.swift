import Metal
import Foundation
import UIKit
import Accelerate

protocol SpectrogramRendererDataSource: class {
    var texture: MTLTexture { get }
    var textureHeightOffset: Int { get }
}


class SpectrogramRenderer {

    private let commandQueue: MTLCommandQueue

    private let renderPipelineState: MTLRenderPipelineState

    private let vertexBuffer: MTLBuffer

    let device: MTLDevice

    weak var dataSource: SpectrogramRendererDataSource?

    /// Current zoom state 
    var zoom: Float = 1.0

    convenience init() {

        let device = MTLCreateSystemDefaultDevice()!

        let dataSize = Self.rectangleVertices.count * MemoryLayout.size(ofValue: Self.rectangleVertices[0])
        guard let vertexBuffer = device.makeBuffer(bytes: Self.rectangleVertices, length: dataSize, options: []) else {
            fatalError("Could not allocate the required memory on the GPU")
        }

        let renderPipelineState = Self.createRenderPipelineState(for:device)

        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Unable to create a new GPU command queue")
        }

        self.init(
            commandQueue: commandQueue,
            renderPipelineState: renderPipelineState,
            vertexBuffer: vertexBuffer,
            device: device
        )
    }

    init(
        commandQueue: MTLCommandQueue,
        renderPipelineState: MTLRenderPipelineState,
        vertexBuffer: MTLBuffer,
        device: MTLDevice
    ) {
        self.commandQueue = commandQueue
        self.renderPipelineState = renderPipelineState
        self.vertexBuffer = vertexBuffer
        self.device = device
    }

    func render(drawable: CAMetalDrawable) {

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)

        guard
            let dataSource = self.dataSource,
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        else { return }

        renderEncoder.setRenderPipelineState(renderPipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentTexture(dataSource.texture, index: 0)

        var params = FragmentShaderParameters(zoom: zoom, textureHeightOffset:smoothedNormalizedTextureHeightOffset)
        renderEncoder.setFragmentBytes(&params, length: MemoryLayout<FragmentShaderParameters>.stride, index: 0)

        renderEncoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: Self.rectangleVertices.count / 4,
            instanceCount: 1
        )
        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private var previousTextureHeightOffset: Int = 0
    private var previousTextureHeightReceivedAt: Double = CACurrentMediaTime()
    private var latestTextureHeightOffset: Int = 0
    private var latestTextureHeightReceivedAt: Double = CACurrentMediaTime()

    /// Because we only get new microphone data every 100ms we have a smooth texture offset so the
    /// texture still translates nicely across the screen without stutters
    private var smoothedNormalizedTextureHeightOffset: Float {
        guard let dataSource = self.dataSource else { return 0.0 }

        let textureHeight = Float(dataSource.texture.height)

        // Extrapolate offset
        if (dataSource.textureHeightOffset == latestTextureHeightOffset) {

            //let timeDiff = Float(latestTextureHeightReceivedAt - previousTextureHeightReceivedAt)
            let timeDiff: Float = 0.1
            let offsetDiff = Float(latestTextureHeightOffset - previousTextureHeightOffset)

            let timeSinceLatest = Float(CACurrentMediaTime() - latestTextureHeightReceivedAt)
            let extrapolatedOffset: Float = timeSinceLatest * offsetDiff / timeDiff

            return (Float(latestTextureHeightOffset) + extrapolatedOffset).truncatingRemainder(dividingBy: textureHeight) / textureHeight
        }

        // Update with new offset
        previousTextureHeightOffset = latestTextureHeightOffset
        previousTextureHeightReceivedAt = latestTextureHeightReceivedAt
        latestTextureHeightOffset = dataSource.textureHeightOffset
        latestTextureHeightReceivedAt = CACurrentMediaTime()

        return Float(latestTextureHeightOffset).truncatingRemainder(dividingBy: textureHeight) / textureHeight
    }

    // MARK: -

    // Create simple rectangle from two triangles to draw
    // the texture onto
    static let rectangleVertices: [Float] = [
        -1.0, -1.0, 0.0, 1.0,
         1.0, -1.0, 0.0, 1.0,
        -1.0,  1.0, 0.0, 1.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0, -1.0, 0.0, 1.0,
         1.0,  1.0, 0.0, 1.0
    ]

    static private func createRenderPipelineState(for device: MTLDevice) -> MTLRenderPipelineState
    {
        let (fragmentShader, vertexShader) = Self.shaders(for: device)

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexShader
        descriptor.fragmentFunction = fragmentShader
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        return try! device.makeRenderPipelineState(descriptor: descriptor)
    }

    static private func shaders(for device: MTLDevice) -> (
        fragmentShader: MTLFunction,
        vertexShader: MTLFunction
    ) {
        guard
            let defaultLibrary = device.makeDefaultLibrary(),
            let fragmentShader = defaultLibrary.makeFunction(name: "fragment_shader"),
            let vertexShader = defaultLibrary.makeFunction(name: "vertex_shader")
        else {
            fatalError("Could not create GPU code library")
        }

        return (fragmentShader, vertexShader)
    }

    func createMetalLayer(frame: CGRect) -> CAMetalLayer {
        let metalLayer = CAMetalLayer()
        metalLayer.device = device
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = true
        metalLayer.contentsScale = UIScreen.main.scale
        metalLayer.frame = frame
        return metalLayer
    }
}

/// Struct that corresponds exactly with the parameters struct in the shader
fileprivate struct FragmentShaderParameters {
    let zoom: Float
    let textureHeightOffset: Float
}
