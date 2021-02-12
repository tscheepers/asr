import Foundation
import Accelerate

/// Numeric calculation optimizations using vDSP (for the vDSP FFT code check Matrix+FFT.swift)

// MARK: - Float
extension Matrix where T == Float {

    // PLUS
    static func + (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vadd(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func + (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs
        lhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = rhs
                vDSP_vsadd(src.baseAddress!, 1, &scalar, dst.baseAddress!, 1, vDSP_Length(lhs.count))
            }
        }
        return result
    }
    static func + (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs + lhs
    }

    // MINUS
    static func - (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vsub(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func - (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        return lhs + (-rhs)
    }
    static func - (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        var result = rhs
        result.flat.withUnsafeMutableBufferPointer{ dst in
            var scalar = lhs
            let length = vDSP_Length(rhs.count)
            vDSP_vneg(dst.baseAddress!, 1, dst.baseAddress!, 1, length)
            vDSP_vsadd(dst.baseAddress!, 1, &scalar, dst.baseAddress!, 1, length)
        }
        return result
    }

    // TIMES
    static func * (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vmul(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func * (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs
        lhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = rhs
                vDSP_vsmul(src.baseAddress!, 1, &scalar, dst.baseAddress!, 1, vDSP_Length(lhs.count))
            }
        }
        return result
    }
    static func * (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs * lhs
    }

    // DIVIDE
    static func / (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vdiv(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func / (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs
        lhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = rhs
                vDSP_vsdiv(src.baseAddress!, 1, &scalar, dst.baseAddress!, 1, vDSP_Length(lhs.count))
            }
        }
        return result
    }
    static func / (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        var result = rhs
        rhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = lhs
                vDSP_svdiv(&scalar, src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(rhs.count))
            }
        }
        return result
    }
}

/// Power function
func pow(_ x: Matrix<Float>, power: Float) -> Matrix<Float> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            if power == 2 {
                vDSP_vsq(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(x.width * x.height))
            } else {
                var size = Int32(x.width * x.height)
                var exponent = power
                vvpowsf(dst.baseAddress!, &exponent, src.baseAddress!, &size)
            }
        }
    }
    return result
}

/// Square root function
func sqrt(_ x: Matrix<Float>) -> Matrix<Float> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            var size = Int32(x.width * x.height)
            vvsqrtf(dst.baseAddress!, src.baseAddress!, &size)
        }
    }
    return result
}

/// Exponential function
func exp(_ x: Matrix<Float>) -> Matrix<Float> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            var size = Int32(x.width * x.height)
            vvexpf(dst.baseAddress!, src.baseAddress!, &size)
        }
    }
    return result
}

/// Natural logarithm
func log(_ x: Matrix<Float>) -> Matrix<Float> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            var size = Int32(x.width * x.height)
            vvlogf(dst.baseAddress!, src.baseAddress!, &size)
        }
    }
    return result
}

// MARK: - Double
extension Matrix where T == Double {

    // PLUS
    static func + (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vaddD(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func + (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs
        lhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = rhs
                vDSP_vsaddD(src.baseAddress!, 1, &scalar, dst.baseAddress!, 1, vDSP_Length(lhs.count))
            }
        }
        return result
    }
    static func + (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs + lhs
    }

    // MINUS
    static func - (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vsubD(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func - (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        return lhs + (-rhs)
    }
    static func - (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        var result = rhs
        result.flat.withUnsafeMutableBufferPointer{ dst in
            var scalar = lhs
            let length = vDSP_Length(rhs.count)
            vDSP_vnegD(dst.baseAddress!, 1, dst.baseAddress!, 1, length)
            vDSP_vsaddD(dst.baseAddress!, 1, &scalar, dst.baseAddress!, 1, length)
        }
        return result
    }

    // TIMES
    static func * (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vmulD(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func * (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs
        lhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = rhs
                vDSP_vsmulD(src.baseAddress!, 1, &scalar, dst.baseAddress!, 1, vDSP_Length(lhs.count))
            }
        }
        return result
    }
    static func * (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        return rhs * lhs
    }

    // DIVIDE
    static func / (lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        assert(lhs.shape == rhs.shape, "Shapes invalid")
        var result = Matrix.zeros(shape: lhs.shape)
        rhs.flat.withUnsafeBufferPointer{ srcR in
            lhs.flat.withUnsafeBufferPointer{ srcL in
                result.flat.withUnsafeMutableBufferPointer{ dst in
                    vDSP_vdivD(srcR.baseAddress!, 1, srcL.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    static func / (lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        var result = lhs
        lhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = rhs
                vDSP_vsdivD(src.baseAddress!, 1, &scalar, dst.baseAddress!, 1, vDSP_Length(lhs.count))
            }
        }
        return result
    }
    static func / (lhs: T, rhs: Matrix<T>) -> Matrix<T> {
        var result = rhs
        rhs.flat.withUnsafeBufferPointer{ src in
            result.flat.withUnsafeMutableBufferPointer{ dst in
                var scalar = lhs
                vDSP_svdivD(&scalar, src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(rhs.count))
            }
        }
        return result
    }
}

/// Power function
func pow(_ x: Matrix<Double>, power: Double) -> Matrix<Double> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            if power == 2 {
                vDSP_vsqD(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(x.width * x.height))
            } else {
                var size = Int32(x.width * x.height)
                var exponent = power
                vvpows(dst.baseAddress!, &exponent, src.baseAddress!, &size)
            }
        }
    }
    return result
}

/// Square root function
func sqrt(_ x: Matrix<Double>) -> Matrix<Double> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            var size = Int32(x.width * x.height)
            vvsqrt(dst.baseAddress!, src.baseAddress!, &size)
        }
    }
    return result
}

/// Exponential function
func exp(_ x: Matrix<Double>) -> Matrix<Double> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            var size = Int32(x.width * x.height)
            vvexp(dst.baseAddress!, src.baseAddress!, &size)
        }
    }
    return result
}

/// Natural logarithm
func log(_ x: Matrix<Double>) -> Matrix<Double> {
    var result = x
    x.flat.withUnsafeBufferPointer { src in
        result.flat.withUnsafeMutableBufferPointer { dst in
            var size = Int32(x.width * x.height)
            vvlog(dst.baseAddress!, src.baseAddress!, &size)
        }
    }
    return result
}
