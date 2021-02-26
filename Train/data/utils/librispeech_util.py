#!/usr/bin/python

if __name__ == '__main__':

    input_outputs = [
        ('/home/thijs/Datasets/LibriSpeech/dev_transcriptions.tsv', '/home/thijs/Datasets/LibriSpeech/prepared_dev_transcriptions.tsv'),
        ('/home/thijs/Datasets/LibriSpeech/test_transcriptions.tsv', '/home/thijs/Datasets/LibriSpeech/prepared_test_transcriptions.tsv'),
        ('/home/thijs/Datasets/LibriSpeech/train_transcriptions.tsv', '/home/thijs/Datasets/LibriSpeech/prepared_train_transcriptions.tsv')
    ]

    for (input, output) in input_outputs:
        with open(input, 'r') as r:
            r.readline().strip('\n')  # Skip the first line
            with open(output, 'w', newline='') as w:
                for i, line in enumerate(r.readlines()):
                    if i != 0:
                        w.write('\n')
                    split = line.strip('\n').split('\t')
                    w.write(split[0] + '\t' + split[2])
