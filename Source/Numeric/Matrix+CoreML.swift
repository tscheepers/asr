import CoreML
import Foundation

extension Matrix where T == Float {

    func createMLMultiArray(prependDimensions: Int = 0) -> MLMultiArray {
        let shape = ([Int](repeating: 1, count: prependDimensions) + [height, width]).map { NSNumber(value: $0) }
        let multiArray = try! MLMultiArray(shape: shape, dataType: .float32)
        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<multiArray.count {
            ptr.advanced(by: i).initialize(to: self.flat[i])
        }
        return multiArray
    }

}

extension MLMultiArray {

    /// This will concatenate all dimensions into one one-dim array.
    func toFloatArray() -> [Float] {
        guard case .float32 = self.dataType else {
            fatalError("This is not a float32 MLMultiArray")
        }

        var arr: [Float] = Array(repeating: 0, count: self.count)
        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(self.dataPointer))
        for i in 0..<self.count {
            arr[i] = Float(ptr[i])
        }
        return arr
    }

    /// This will create a matrix
    func toFloatMatrix() -> Matrix<Float> {
        guard case .float32 = self.dataType else {
            fatalError("This is not a float32 MLMultiArray")
        }

        let shape = self.shape.map({ $0.intValue }).filter({ $0 > 1 })

        guard shape.count == 2 else {
            fatalError("The dimensions of this MLMultiArray do not allow it to be converted into a matrix")
        }

        return Matrix(shape: (shape[0], shape[1]), flat: self.toFloatArray())
    }
}
