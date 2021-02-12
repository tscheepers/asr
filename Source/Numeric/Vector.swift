import Foundation

extension Array where Element: Comparable {

    /// Index of the maximum value in the array
    func argmax() -> Index? {
        return indices.max(by: { self[$0] < self[$1] })
    }

    /// Index of the minimum value in the array
    func argmin() -> Index? {
        return indices.min(by: { self[$0] < self[$1] })
    }
}

extension Array {
    /// Creates a new Array by repeating an existing array
    init(repeating: [Element], count: Int) {
        self.init([[Element]](repeating: repeating, count: count).flatMap{$0})
    }

    /// New Array from current array by repeating it `count` times
    func repeated(count: Int) -> [Element] {
        return [Element](repeating: self, count: count)
    }
}
