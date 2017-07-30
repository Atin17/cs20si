import json
import string

from nltk import word_tokenize

DATA_PATH = 'extractor_dump.json'
OUTPUT_PATH = 'enwik9_cleaned.txt'
PUNCTUATION = set(string.punctuation).union(['``', "''", '...'])


def read_data(data_path):
    with open(data_path, 'r') as f:
        for line in f:
            yield json.loads(line.decode('utf-8'))['text'].strip()

def clean_data(data_path):
    data = read_data(data_path)
    with open(OUTPUT_PATH, 'w+') as f:
        for text in data:
            for token in clean_text(text):
                f.write((token + '\n').encode('utf-8'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in PUNCTUATION]


if __name__ == '__main__':
    clean_data(DATA_PATH)

