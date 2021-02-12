import XCTest

class ComplexMatrixTest: XCTestCase {

    func testMagnitude() {

        let result = ComplexMatrix(
            shape: (Fixtures.stftOutput.count, Fixtures.stftOutput[0].count),
            flatReal: Fixtures.stftOutput.flatMap({ $0.map({ $0.real }) }),
            flatImag: Fixtures.stftOutput.flatMap({ $0.map({ $0.imag }) })
        ).magnitude()

        ASRAssertEqual(result, Fixtures.magnitudesOutput, accuracy: 0.01)
    }

}
