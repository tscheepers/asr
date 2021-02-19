# Dunglish (Dutch & English) Automated Speech Recognition for the iPhone

With this project I am trying to create an iPhone app that demonstrates streaming on-device speech recognition for inputs that blend the Dutch and English language. 

Training data is gathered from [Librispeech](http://www.openslr.org/12/), [Common Voice](https://commonvoice.mozilla.org/) and [Corpus Gesproken Nederlands](https://taalmaterialen.ivdnt.org/download/tstc-corpus-gesproken-nederlands/).

The model's current architecture is mostly inspired by the original [DeepSpeech](https://arxiv.org/abs/1412.5567) paper, and written using [PyTorch](https://pytorch.org/). The PyTorch model is to be converted using [coremltools](https://github.com/apple/coremltools).

This project is a work in progress.

Training:
- [x] Write ASR model ready for streaming inference, with lookahead, without bi-directional LSTM
- [x] Implement streaming inference proof-of-concept
- [x] Train on LibriSpeech (English)
- [x] Prepare Corpus Gesproken Nederlands, _thanks [@wilrop](https://github.com/wilrop/Import-CGN)_  
- [ ] Train on Common Voice + Corpus Gesproken Nederlands
- [ ] Train a small Transformer language model on Dutch Wikipedia, to improve decoding
- [x] Integrate spec augment
- [ ] Create a second model class using 3xFC->1xLSTM->1xFC and windowed input, inspired by the up-to-date model by [Mozilla](https://github.com/mozilla/DeepSpeech)
- [ ] Improve spec augment with additions done by [Mozilla](https://github.com/mozilla/DeepSpeech)
- [ ] Train model on the Dutch and English datasets simultaniously to create a blended model that is able to predict both languages

Inference:
- [x] Write custom Short-Time Fourier Transform for iPhone [_[code](Source/Numeric/Vector+STFT.swift)_]
- [x] Test our custom STFT with [librosa](https://librosa.org/) reference
- [x] Load pretrained [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) using coremltools
- [ ] Load model reading for streaming inference
- [ ] Implement streaming inference in Swift
- [x] Implement greedy decoding in Swift
- [ ] Implement CTC Beam Search decoder in Swift
- [ ] Use microphone for streaming inference
- [x] Load custom model using coremltools into the iPhone codebase
- [ ] Augment decoder with Transformer based language model
- [ ] Render spectrogram input as it is comming in from the microphone
- [ ] Render raw model output characters on top of the spectrogram
- [ ] Render decoded model output below the raw output
