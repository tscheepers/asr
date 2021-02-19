import re
from config import Config


class StringProcessor:
    def __init__(self, config: Config):
        self.lowercase = config.lowercase
        self.chars = [c for c in config.valid_chars]

        if len(config.blank_char) != 1:
            raise Exception("The blank_char configuration setting should be a single character.")

        blank_char = config.blank_char[0]
        if blank_char in self.chars:
            raise Exception("The blank character cannot be a part of the valid characters.")

        # Add the blank token
        self.blank_id = len(self.chars)
        self.chars += [blank_char]

        self.char_to_idx = {char: idx for (idx, char) in enumerate(self.chars)}

    def str_to_labels(self, string):

        if self.lowercase:  # to lower case
            string = string.lower()

        # remove all chars not in the char list
        string = ''.join(c for c in string if c in self.char_to_idx and self.char_to_idx[c] != self.blank_id)

        # remove double spaces
        string = re.sub(' +', ' ', string)

        # remove leading and trailing spaces
        string = string.strip()

        return [self.char_to_idx[char] for char in string]

    def labels_to_str(self, labels, split_every=None):
        result = ''.join([self.chars[int(idx)] for idx in labels])

        # Insert a "|" every n characters
        if split_every is not None:
            return '|'.join(result[i:i + split_every] for i in range(0, len(result), split_every))

        return result


