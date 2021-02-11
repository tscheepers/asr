import Accelerate
import Foundation


// MARK: - DFT

extension Matrix where T: FloatingPoint {

    /// Calculate the Discrete Fourier Transform
    func rowWiseDiscreteFourierTransform() -> ComplexMatrix<T> {
        switch self {
        case let matrix as Matrix<Float>:
            return matrix.rowWiseDiscreteFourierTransform() as! ComplexMatrix<T>
        case let matrix as Matrix<Double>:
            return matrix.rowWiseDiscreteFourierTransform() as! ComplexMatrix<T>
        default:
            fatalError("rowWiseDiscreteFourierTransform() for this type is not implemented")
        }
    }
}

extension Matrix where T == Float {

    func rowWiseDiscreteFourierTransform() -> ComplexMatrix<Float> {
        var flatReals: [Float] = []
        var flatImags: [Float] = []
        var reuseSetup: vDSP_DFT_Setup? = nil

        for i in (0..<self.height) {
            let (reals, imags, fftSetup) = discreteFourierTransform(reals: self[i], reuseSetup: reuseSetup)
            flatReals += reals
            flatImags += imags
            reuseSetup = fftSetup
        }

        if let setupToDestroy = reuseSetup {
            vDSP_DFT_DestroySetup(setupToDestroy)
        }

        return ComplexMatrix<Float>(shape: (height: self.shape.height, width: self.width/2+1), flatReal: flatReals, flatComplex: flatImags)
    }
}

fileprivate func discreteFourierTransform(reals: [Float], imags: [Float]? = nil, reuseSetup: vDSP_DFT_Setup? = nil) -> ([Float], [Float], vDSP_DFT_Setup) {

    let realsIn: [Float] = reals
    let imagsIn: [Float] = imags ?? [Float](repeating: 0.0, count: reals.count)

    guard realsIn.count == imagsIn.count else {
        fatalError("You should have the samen number or real and immaginary values")
    }

    guard let dftSetup = vDSP_DFT_zop_CreateSetup(reuseSetup, vDSP_Length(reals.count), .FORWARD) else {
        fatalError("Could not initialize DFT setup. Perhapse the number or values does not equal to: f * 2**n, where f is 1, 3, 5, or 15 and n is at least 3.")
    }

    var realsOut = [Float](repeating: 0.0, count: reals.count)
    var imagsOut = [Float](repeating: 0.0, count: reals.count)

    vDSP_DFT_Execute(dftSetup, realsIn, imagsIn, &realsOut, &imagsOut)

    // When the DFT is computed for purely real input, the output is Hermitian-symmetric,
    // i.e. the negative frequency terms are just the complex conjugates of the corresponding
    // positive-frequency terms, and the negative-frequency terms are therefore redundant.
    if imags == nil {
        realsOut = Array(realsOut[0..<reals.count/2+1])
        imagsOut = Array(realsOut[0..<reals.count/2+1])
    }

    return (realsOut, imagsOut, dftSetup)
}

extension Matrix where T == Double {

    func rowWiseDiscreteFourierTransform() -> ComplexMatrix<Double> {
        var flatReals: [Double] = []
        var flatImags: [Double] = []
        var reuseSetup: vDSP_DFT_SetupD? = nil

        for i in (0..<self.height) {
            let (reals, imags, fftSetup) = discreteFourierTransform(reals: self[i], reuseSetup: reuseSetup)
            flatReals += reals
            flatImags += imags
            reuseSetup = fftSetup
        }

        if let setupToDestroy = reuseSetup {
            vDSP_DFT_DestroySetupD(setupToDestroy)
        }

        return ComplexMatrix<Double>(shape: (height: self.shape.height, width: self.width/2+1), flatReal: flatReals, flatComplex: flatImags)
    }
}

fileprivate func discreteFourierTransform(reals: [Double], imags: [Double]? = nil, reuseSetup: vDSP_DFT_SetupD? = nil) -> ([Double], [Double], vDSP_DFT_SetupD) {

    let realsIn = reals
    let imagsIn = imags ?? [Double](repeating: 0.0, count: reals.count)

    guard realsIn.count == imagsIn.count else {
        fatalError("You should have the samen number or real and immaginary values")
    }

    guard let dftSetup = vDSP_DFT_zop_CreateSetupD(reuseSetup, vDSP_Length(reals.count), .FORWARD) else {
        fatalError("Could not initialize DFT setup. Perhapse the number or values does not equal to: f * 2**n, where f is 1, 3, 5, or 15 and n is at least 3.")
    }

    var realsOut = [Double](repeating: 0.0, count: reals.count)
    var imagsOut = [Double](repeating: 0.0, count: reals.count)

    vDSP_DFT_ExecuteD(dftSetup, realsIn, imagsIn, &realsOut, &imagsOut)

    // When the DFT is computed for purely real input, the output is Hermitian-symmetric,
    // i.e. the negative frequency terms are just the complex conjugates of the corresponding
    // positive-frequency terms, and the negative-frequency terms are therefore redundant.
    if imags == nil {
        realsOut = Array(realsOut[0..<reals.count/2+1])
        imagsOut = Array(imagsOut[0..<reals.count/2+1])
    }

    return (realsOut, imagsOut, dftSetup)
}
