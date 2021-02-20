# Dunglish (Dutch & English) Streaming Automatic Speech Recognition for the iPhone

With this project I am trying to create an iPhone app that demonstrates streaming on-device automatic speech recognition for inputs that blend the Dutch and English language. 
The current model architecture is inspired by the [DeepSpeech 2](https://arxiv.org/abs/1512.02595) paper. 

**This project is a work in progress.**

### Dependencies

The iPhone app is written in Swift 5.3.2 using Xcode 12.4 for iOS 14.4.
The training is code written using Python 3.8.7 with PyTorch 1.7.1, cuda 11.0.x and cuDNN 8.0.x.
This code also has [additional dependencies](Train/requirements.txt) including [ctcdecode](https://github.com/parlance/ctcdecode) which needs to be installed by hand.
Model conversion is done using [coremltools](https://github.com/apple/coremltools) 4.1.

### Training data
I used training data  gathered from:
 - [Librispeech](http://www.openslr.org/12/); 
 - [Common Voice](https://commonvoice.mozilla.org/) (Dutch and English); and
 - [Corpus Gesproken Nederlands](https://taalmaterialen.ivdnt.org/download/tstc-corpus-gesproken-nederlands/) (processed using [Import CGN](https://github.com/wilrop/Import-CGN)).

Prepare the training data by creating three tab separated (`.tsv`) files.
These files should have two columns. The **first column** should contain the full filesystem path to a sound file.
These can be in the in a format [librosa](https://librosa.org/) supports, e.g. `.wav`, `.mp3` or `.flac`.
The **second column** should contain the entire sentence. Each line in the file is a new sample.

Example `train.tsv` file:
```
/mnt/datadisk/datasets/mydataset/louis-1.mp3	We are running after de feiten aan.
/mnt/datadisk/datasets/mydataset/assistent.mp3	Hey GoLexIri, zet One Flew Over the Cuckoo's Nest aan op de TV in de woonkamer.
/mnt/datadisk/datasets/mydataset/snel-fix.mp3	Laten we dat alsjeblieft fixen, as soon as possible.
/mnt/datadisk/datasets/mydataset/kat-wijs.mp3	Maak dat the cat wise.
```

### Training to-dos: 
- [x] Write ASR model ready for streaming inference, uni-directional LSTM with lookahead  [`code`](Train/model/cnn_rnn_lookahead_acoustic_model.py)
- [x] Implement streaming inference proof-of-concept [`code`](Train/streaming_inference.py)
- [x] Train on LibriSpeech (English)
- [x] Export the model using coremltools [`code`](Train/coreml.py)
- [x] Prepare Corpus Gesproken Nederlands, _thanks [@wilrop](https://github.com/wilrop/Import-CGN)_  
- [ ] Train on Common Voice Dutch + Corpus Gesproken Nederlands
- [ ] Train on LibriSpeech + Common Voice + Corpus Gesproken Nederlands
- [ ] Train a small Transformer language model on Dutch Wikipedia, to improve decoding
- [x] Integrate spec augment
- [ ] Create a second model class using `3 FC` → `1 LSTM` → `1 FC` and windowed input, inspired by the up-to-date model by [Mozilla](https://github.com/mozilla/DeepSpeech)
- [ ] Improve spec augment with additions done by [Mozilla](https://github.com/mozilla/DeepSpeech)
- [ ] Train model on the Dutch and English datasets simultaniously to create a blended model that is able to predict both languages

### Inference to-dos:
- [x] Write custom Short-Time Fourier Transform for iPhone [`code`](Source/Numeric/Vector+STFT.swift)
- [x] Test our custom STFT with [librosa](https://librosa.org/) reference
- [x] Load pretrained [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) using coremltools
- [ ] Load model reading for streaming inference
- [ ] Implement streaming inference in Swift
- [x] Implement greedy decoding in Swift
- [ ] Implement CTC Beam Search decoder in Swift
- [ ] Use microphone for streaming inference
- [x] Load custom model using coremltools into the iPhone codebase
- [ ] Augment decoder with Transformer based language model
- [ ] Render spectrogram input as it is coming in from the microphone
- [ ] Render raw model output characters on top of the spectrogram
- [ ] Render decoded model output below the raw output
