import XCTest

class VectorTests: XCTestCase {

    func testPadWithReflection() throws {
        let input = (1...5).map { Float($0) }
        let result = input.padWithReflection(3)
        let expected: [Float] = [4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
        XCTAssertEqual(result, expected)
    }

    func testFramesFromSlidingWindow() throws {
        let input = (1...8).map { Float($0) }
        let result = input.framesFromSlidingWindow(windowLength: 4, hopLength: 2)
        let expected: [[Float]] = [[1.0, 2.0, 3.0, 4.0],
                                   [3.0, 4.0, 5.0, 6.0],
                                   [5.0, 6.0, 7.0, 8.0]]
        ASRAssertEqual(result, expected)
    }

    func testInterpolation() {
        let result = [Float].interpolation(start: 1, stop: 2, count: 5)
        let expected: [Float] = [1.0, 1.25, 1.5, 1.75, 2.0]
        XCTAssertEqual(result, expected)
    }

    func testHannWindow() {
        let result = [Float].hannWindow(windowLength: 9)
        let expected: [Float] = [0.0, 0.15, 0.5, 0.85, 1.0, 0.85, 0.5, 0.15, 0.0]

        for i in 0..<result.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 0.01)
        }
    }

    func testSTFT() {
        let input = Fixtures.libriSpeechSample
        XCTAssertEqual(input.count, 191280)

        let result = input.shortTimeFourierTransform(nFFT: 320, hopLength: 160)
        XCTAssertEqual(result.width, 161)
        XCTAssertEqual(result.height, 1196)

        ASRAssertEqual(result, Fixtures.stftOutput, accuracy: 0.01)
    }

    func testSTFTDouble() {
        let input = Fixtures.libriSpeechSample.map { Double($0) }
        XCTAssertEqual(input.count, 191280)

        let result = input.shortTimeFourierTransform(nFFT: 320, hopLength: 160)
        XCTAssertEqual(result.width, 161)
        XCTAssertEqual(result.height, 1196)

        let expected: [[Complex<Double>]] = Fixtures.stftOutput.map { $0.map { Complex<Double>(Double($0.real), Double($0.imag)) } }

        ASRAssertEqual(result, expected, accuracy: 0.01)
    }

}

