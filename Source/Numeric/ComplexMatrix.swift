import Foundation

typealias Complex<T> = (real: T, imag: T)

// MARK: - Matrix
struct ComplexMatrix<T> {

    var shape: (height: Int, width: Int)
    var height: Int { return shape.0 }
    var width: Int { return shape.1 }

    var flatReal: [T]
    var flatComplex: [T]

    init(shape: (height: Int, width: Int), flatReal: [T], flatComplex: [T]) {
        assert(flatReal.count == shape.height * shape.width, "Flat does not correspond with shape")
        assert(flatComplex.count == shape.height * shape.width, "Flat does not correspond with shape")
        self.shape = shape
        self.flatReal = flatReal
        self.flatComplex = flatComplex
    }

    /// Access the matrix using `matrix[x,y]`
    subscript(row: Int, col: Int) -> Complex<T> {
        get {
            assert(row >= 0 && row < height, "Row index out of range")
            assert(col >= 0 && col < width, "Col index out of range")
            return (self.flatReal[(row * width) + col], self.flatComplex[(row * width) + col])
        }
        set {
            assert(row >= 0 && row < height, "Row index out of range")
            assert(col >= 0 && col < width, "Col index out of range")
            self.flatReal[(row * width) + col] = newValue.real
            self.flatComplex[(row * width) + col] = newValue.imag
        }
    }

    func map<A>(_ transform: (Complex<T>) throws -> A) rethrows -> Matrix<A> {
        return Matrix<A>(shape: shape, flat: try zip(self.flatReal, self.flatComplex).map(transform))
    }
}

// MARK: - Debug
extension ComplexMatrix: CustomDebugStringConvertible {

    /// Printing out the matrix in a nice readable format
    var debugDescription: String {
        var string = "height: \(height) width: \(width)\n ["
        for row in 0..<height {
            string.append("[")
            for col in 0..<width {
                string.append("\(self[row, col].real) + \(self[row, col].imag)i")
                if col != width - 1 {
                    string.append(", ")
                }
            }
            string.append("]\n")
        }
        string.append("]")
        return string
    }
}
