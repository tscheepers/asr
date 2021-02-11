import XCTest

func ASRAssertEqual<T>(_ matrix: Matrix<T>, _ list: [[T]]) where T : FloatingPoint {
    for i in 0..<matrix.height {
        for j in 0..<matrix.width {
            XCTAssertEqual(matrix[i, j], list[i][j], "(\(i),\(j)) not equal")
        }
    }
}

func ASRAssertEqual<T>(_ matrix: ComplexMatrix<T>, _ list: [[Complex<T>]], accuracy: T) where T : FloatingPoint {
    for i in 0..<matrix.height {
        for j in 0..<matrix.width {
            XCTAssertEqual(matrix[i, j].real, list[i][j].real, accuracy: accuracy, "(\(i),\(j)) real not equal")
            XCTAssertEqual(matrix[i, j].imag, list[i][j].imag, accuracy: accuracy, "(\(i),\(j)) imaginary not equal")
        }
    }
}
