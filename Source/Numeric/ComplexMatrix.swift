import Foundation

typealias Complex<T> = (real: T, imag: T)

// MARK: - Matrix
struct ComplexMatrix<T> {

    var shape: (height: Int, width: Int)
    var height: Int { return shape.0 }
    var width: Int { return shape.1 }
    var count: Int { return width * height }

    var flatReal: [T]
    var flatImag: [T]

    init(shape: (height: Int, width: Int), flatReal: [T], flatImag: [T]) {
        assert(flatReal.count == shape.height * shape.width, "Flat does not correspond with shape")
        assert(flatImag.count == shape.height * shape.width, "Flat does not correspond with shape")
        self.shape = shape
        self.flatReal = flatReal
        self.flatImag = flatImag
    }

    /// Access the matrix using `matrix[x,y]`
    subscript(row: Int, col: Int) -> Complex<T> {
        get {
            assert(row >= 0 && row < height, "Row index out of range")
            assert(col >= 0 && col < width, "Col index out of range")
            return (self.flatReal[(row * width) + col], self.flatImag[(row * width) + col])
        }
        set {
            assert(row >= 0 && row < height, "Row index out of range")
            assert(col >= 0 && col < width, "Col index out of range")
            self.flatReal[(row * width) + col] = newValue.real
            self.flatImag[(row * width) + col] = newValue.imag
        }
    }

    func map<A>(_ transform: (Complex<T>) throws -> A) rethrows -> Matrix<A> {
        return Matrix<A>(shape: shape, flat: try zip(self.flatReal, self.flatImag).map(transform))
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

// MARK: - Debug
extension ComplexMatrix where T: FloatingPoint {

    /// A matrix containing the absolute value of each element in x. For complex input, a + ib, the absolute value is \sqrt{ a^2 + b^2 }.
    func absolute() -> Matrix<T> {
        return self.map { sqrt($0.real * $0.real + $0.imag * $0.imag) }
    }

    /// The magnitude for each element in the matrix
    func magnitude() -> Matrix<T> {
        return self.absolute()
    }
}
