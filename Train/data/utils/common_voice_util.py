#!/usr/bin/python


if __name__ == '__main__':

    def data_dir(*args, root="/home/thijs/Datasets/cv-corpus-6.1-2020-12-11/nl"):
        return "/".join([root] + list(args))

    input_outputs = [
        (data_dir('dev.tsv'), data_dir('prepared_dev.tsv')),
        (data_dir('test.tsv'), data_dir('prepared_test.tsv')),
        (data_dir('train.tsv'), data_dir('prepared_train.tsv')),
    ]

    for (input, output) in input_outputs:
        with open(input, 'r') as r:
            r.readline()  # Skip the first line
            with open(output, 'w') as w:
                for i, line in enumerate(r.readlines()):
                    if i != 0:
                        w.write('\n')
                    split = line.split('\t')
                    w.write(data_dir('clips', split[1]) + '\t' + split[2])
