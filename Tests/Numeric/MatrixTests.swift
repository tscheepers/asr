import XCTest
import CoreML

class MatrixTests: XCTestCase {
    
    func testMagnitudeLog1p() {

        let height = Fixtures.magnitudesOutput.count
        let width = Fixtures.magnitudesOutput[0].count
        let magnitude = Matrix(shape: (height, width), flat: Fixtures.magnitudesOutput.flatMap({ $0 }))

        let log1p = log(magnitude + 1)

        ASRAssertEqual(log1p, Fixtures.log1POutput, accuracy: 0.01)
    }

    func testNormalizedSpectrogram() {
        let height = Fixtures.log1POutput.count
        let width = Fixtures.log1POutput[0].count
        let spectrogram = Matrix(shape: (height, width), flat: Fixtures.log1POutput.flatMap({ $0 }))

        let normalized = (spectrogram - spectrogram.mean) / spectrogram.std

        ASRAssertEqual(normalized, Fixtures.normalizedSpectrogram, accuracy: 0.01)
    }

    func testCoreMLMultiArray() {
        let matrix = Matrix(
            shape: (Fixtures.normalizedSpectrogram.count, Fixtures.normalizedSpectrogram[0].count),
            flat: Fixtures.normalizedSpectrogram.flatMap({ $0 })
        )
        let multiArray = matrix.createMLMultiArray()
        ASRAssertEqual(matrix, multiArray)
    }
}

func ASRAssertEqual<T>(_ matrix: Matrix<T>, _ list: [[T]]) where T : FloatingPoint {
    for i in 0..<matrix.height {
        for j in 0..<matrix.width {
            XCTAssertEqual(matrix[i, j], list[i][j], "(\(i),\(j)) not equal")
        }
    }
}

func ASRAssertEqual(_ matrix: Matrix<Float>, _ multiArray: MLMultiArray) {
    for i in 0..<matrix.height {
        for j in 0..<matrix.width {
            let matrixValue = NSNumber(value: matrix[i, j])
            let multiArrayValue = multiArray[[i,j] as [NSNumber]]
            XCTAssertEqual(matrixValue, multiArrayValue, "(\(i),\(j)) not equal")
        }
    }
}

func ASRAssertEqual<T>(_ matrix: Matrix<T>, _ list: [[T]], accuracy: T) where T : FloatingPoint {
    for i in 0..<matrix.height {
        for j in 0..<matrix.width {
            XCTAssertEqual(matrix[i, j], list[i][j], accuracy: accuracy, "(\(i),\(j)) not equal")
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
