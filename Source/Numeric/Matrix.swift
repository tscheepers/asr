import Foundation


// MARK: - Matrix
struct Matrix<T> {

    var shape: (height: Int, width: Int)
    var height: Int { return shape.0 }
    var width: Int { return shape.1 }
    var count: Int { return width * height }

    var flat: [T]

    init(shape: (height: Int, width: Int), flat: [T]) {
        assert(flat.count == shape.height * shape.width, "Flat does not correspond with shape")
        self.shape = shape
        self.flat = flat
    }

    /// Access the matrix using `matrix[x,y]`
    subscript(row: Int, col: Int) -> T {
        get {
            assert(row >= 0 && row < height, "Row index out of range")
            assert(col >= 0 && col < width, "Col index out of range")
            return self.flat[(row * width) + col]
        }
        set {
            assert(row >= 0 && row < height, "Row index out of range")
            assert(col >= 0 && col < width, "Col index out of range")
            self.flat[(row * width) + col] = newValue
        }
    }

    /// Access a row using `matrix[x]`
    subscript(row: Int) -> [T] {
        get {
            assert(row >= 0 && row < height, "Row index out of range")
            return Array(self.flat[(row * width)..<((row+1) * width)])
        }
    }

    func map<A>(_ transform: (T) throws -> A) rethrows -> Matrix<A> {
        return Matrix<A>(shape: shape, flat: try flat.map(transform))
    }
}

// MARK: - Debug
extension Matrix: CustomDebugStringConvertible {

    /// Printing out the matrix in a nice readable format
    var debugDescription: String {
        var string = "height: \(height) width: \(width)\n ["
        for row in 0..<height {
            string.append("[")
            for col in 0..<width {
                if col == width - 1 {
                    string.append("\(self[row, col])")
                } else {
                    string.append("\(self[row, col]), ")
                }
            }
            string.append("]\n")
        }
        string.append("]")
        return string
    }
}

// MARK: - Float
extension Matrix where T : FloatingPoint {

    /// The sum of the matrix
    var sum: T {
        return flat.reduce(T.zero, { $0 + $1 })
    }

    /// Returns a matrix filled with zeros
    static func zeros(shape: (height: Int, width: Int)) -> Matrix<T> {
        let flat = [T](repeating: T.zero, count: shape.height * shape.width)
        return Matrix<T>(shape: shape, flat: flat)
    }
}

// MARK: - Elementwise operators
extension Matrix where T : AdditiveArithmetic {

    // PLUS
    static func + (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        return Matrix(shape: lhs.shape, flat: zip(lhs.flat, rhs.flat).map({ $0.0 + $0.1 }))
    }
    static func + (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        return lhs.map({ $0 + rhs })
    }
    static func + (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs + lhs
    }

    // MINUS
    static func - (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        return Matrix(shape: lhs.shape, flat: zip(lhs.flat, rhs.flat).map({ $0.0 - $0.1 }))
    }
    static func - (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        return lhs.map({ $0 - rhs })
    }
    static func - (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs.map({ lhs - $0 })
    }
}

extension Matrix where T : Numeric {
    // TIMES
    static func * (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        return Matrix(shape: lhs.shape, flat: zip(lhs.flat, rhs.flat).map({ $0.0 * $0.1 }))
    }
    static func * (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        return lhs.map({ $0 * rhs })
    }
    static func * (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs * lhs
    }
}

extension Matrix where T : FloatingPoint {
    // DIVIDE
    static func / (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        return Matrix(shape: lhs.shape, flat: zip(lhs.flat, rhs.flat).map({ $0.0 / $0.1 }))
    }
    static func / (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        return lhs.map({ $0 / rhs })
    }
    static func / (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs.map({ lhs / $0 })
    }
}
