import Foundation
import XCTest

class ASRTests: XCTestCase {

    func testASR() {
        let asr = ASR()
        measure {
            let text = asr.speechToText(wave: Fixtures.libriSpeechSample)

            // This string is not fully correct of course, but it is correctly decoding the entire file in chuncks
            XCTAssertEqual(text, "in eighteen sixty two a law wis enichty with a purpose of supressing boral merriage ed is had been bredicted in the dashonal citic prayere to its passage it lay fror mitte years had ded lettere")
        }
    }

}
