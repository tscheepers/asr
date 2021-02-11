# Automated Speech Recognition in Dutch for the iPhone

With this project I am trying to create an iPhone app that does on-device speech recognition for the Dutch language from the ground up, i.e. without using any specialized speech recognition libraries.

The model's architecture is mostly inspired by [DeepSpeech](https://arxiv.org/abs/1412.5567), and written using [PyTorch](https://pytorch.org/). Training data is gathered from [Common Voice](https://commonvoice.mozilla.org/) and [Corpus Gesproken Nederlands](https://taalmaterialen.ivdnt.org/download/tstc-corpus-gesproken-nederlands/). The PyTorch model is to be converted using [coremltools](https://github.com/apple/coremltools).

This project is a work in progress.

Training:
- [ ] Write model ready for streaming inference, with lookahead, without bi-directional LSTM
- [ ] Train on LibriSpeech (English)
- [x] Prepare Corpus Gesproken Nederlands, _thanks [@wilrop](https://github.com/wilrop/Import-CGN)_  
- [ ] Train on Common Voice + Corpus Gesproken Nederlands
- [ ] Train a small Transformer language model on Dutch Wikipedia, to improve decoding

Inference:
- [x] Write custom Short-Time Fourier Transform for iPhone [_[code](Source/Numeric/Vector+STFT.swift)_]
- [x] Test our custom STFT with [librosa](https://librosa.org/) reference
- [ ] Load pretrained [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) using coremltools
- [ ] Implement CTC Beam Search decoder in Swift
- [ ] Load custom model using coremltools into the iPhone codebase
- [ ] Augment decoder with Transformer based language model
- [ ] Create fancy UI showing how steaming recognition and beam Search decoding works, while the user is speaking into the phone