import Foundation

extension Array where Element : FloatingPoint {

    /// The Short-time Fourier transform represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
    /// The signal is centered and padded using reflection. Each frame is tapered using the Hann function.
    ///
    /// Similar to: https://librosa.org/doc/latest/generated/librosa.stft.html
    /// Explenation video: https://www.youtube.com/watch?v=8nZrgJjl3wc
    ///
    /// - Parameters:
    ///     - nFFT: Length of the windowed signal. Defaults to `2048`. In any case, we recommend setting `nFFT` to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.
    ///     - hopLength: Number of audio samples between adjacent STFT columns. If unspecified, defaults to `nFFT/4`.
    /// - Returns: The complex matrix with dimensions: `(1 + nFFT/2, nFrames)`
    func shortTimeFourierTransform(nFFT: Int = 2048, hopLength: Int? = nil) -> ComplexMatrix<Element> {

        // We pad both sides with the reflection of the signal, such that the tampering does not remove signal at the edges
        let frames = self.padWithReflection(nFFT/2).framesFromSlidingWindow(windowLength: nFFT, hopLength: hopLength ?? (nFFT/4))

        // Tamper the individual frames to prevent edge artifacts using the Hann function
        let tamperedFrames = frames * [Element].hannWindow(windowLength: nFFT).broadcast(size: frames.height)

        // Apply the discrete fourier transform to each individual row
        return tamperedFrames.rowWiseDiscreteFourierTransform()
    }

    /// Broadcasts the array to a matrix
    func broadcast(size: Int) -> Matrix<Element> {
        return Matrix<Element>(shape: (height: size, width: self.count), flat: self.repeated(count: size))
    }

    /// Pads the outer side of the array as if it reflects inward
    func padWithReflection(_ size: Int) -> [Element] {
        let n = self.count
        if size >= n {
            fatalError("Cannot reflect with a larger size than the array itself.")
        }

        let leftPad = Array(self[1...size].reversed())
        let rightPad = Array(self[(n-size-1)..<n-1].reversed())

        return leftPad + self + rightPad
    }

    /// Create sliding window frames from a one dimensional signal
    func framesFromSlidingWindow(windowLength: Int, hopLength: Int) -> Matrix<Element> {
        let shape = (height: 1 + (self.count - windowLength) / hopLength, width: windowLength)
        let flat: [Element] = (0..<shape.width*shape.height).map { idx in
            return self[(idx % shape.width) + hopLength * (idx / shape.width)]
        }
        return Matrix<Element>.init(shape: shape, flat: flat)
    }

    /// Create an array that contains the interpolated values between start and stop
    /// The first value of the array will be equal to start, the last value will be equal to stop
    static func interpolation(start: Element, stop: Element, count: Int) -> [Element] {

        if count < 2 {
            fatalError("Number of elements should be at least two")
        }

        return (0..<count).map { idx -> Element in
            let idxf = Element(idx)
            let countf = Element(count)
            let x: Element = stop * idxf/(countf - 1)
            let y: Element = start * (1 - idxf/(countf - 1))
            return x + y
        }
    }

    /// Create an array with values from the Hann function
    /// Read: https://en.wikipedia.org/wiki/Hann_function
    static func hannWindow(windowLength: Int) -> [Element] {
        // On an even window length we make sure there is always a peak of 1.0 by adding an extra item
        let evenCorrection = (windowLength % 2 == 0 ? 1 : 0)
        return interpolation(start: 0, stop: 2*Element.pi, count: windowLength + evenCorrection)
            .map { (1 - cos($0))/2 }.dropLast(evenCorrection)

    }
}

/// Cosine for floating point
fileprivate func cos<T: FloatingPoint>(_ x: T) -> T {
    switch x {
    case let v as Double:
        return cos(v) as! T
    case let v as Float:
        return cosf(v) as! T
    default:
        fatalError("Cosine for this type is not implemented")
    }
}
