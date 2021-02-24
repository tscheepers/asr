import Foundation
import Metal
import simd

/// Easy integration with Metal for copying from Textures and filling textures with data from a matrix

// MARK: - Float
extension Matrix where T == Float {

    /// Create a matrix from a texture (float2)
    static func copy(fromTexture texture: MTLTexture) -> Matrix<T> {

        let n = texture.height * texture.width
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: n * MemoryLayout<T>.stride, alignment: MemoryLayout<T>.alignment)
        texture.getBytes(
            pointer,
            bytesPerRow: texture.width * MemoryLayout<T>.stride,
            from: MTLRegionMake2D(0, 0, texture.width, texture.height),
            mipmapLevel: 0
        )

        let typedPointer = pointer.bindMemory(to: Float.self, capacity: n)

        let flat = Array(UnsafeBufferPointer(start: typedPointer, count: n))

        return Matrix<T>(shape: (height: texture.height, width: texture.width), flat: flat)
    }

    /// Fill a metal texture with the matrix
    func fill(texture: MTLTexture) {

        switch texture.pixelFormat {
        case .r32Float:
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: flat,
                bytesPerRow: width * MemoryLayout<T>.stride
            )
        default:
            fatalError("Can not fill a texture with this pixelformat")
        }
    }
}
