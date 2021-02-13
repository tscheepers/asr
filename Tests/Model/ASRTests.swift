import Foundation
import XCTest

class ASRTests: XCTestCase {

    func testASR() {
        let asr = ASR()
        let text = asr.speechToText(wave: Fixtures.libriSpeechSample)
        XCTAssertEqual(text, "IN EIGHTEEN SIXTY TWO ALLAH WAS I NIGTY WITH A PURPOSE OF SUPPRESSING PORAL MARRIAGE AND AS HAD BEEN PREDICTED IN THE NATIONAL SENANT PRAYER TO ITS PASSAGE IT LAY FOR MANY YEARS SAID DEAD LETTER")
    }

}
