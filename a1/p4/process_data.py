import random
from collections import Counter

import numpy as np

import utils

# Parameters for downloading data
FILE_PATH = 'enwik9_cleaned.txt'

def read_data(data_path):
    with open(data_path, 'r') as f:
        for line in f:
            yield line.decode('utf-8').strip()

def build_vocab(vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    print('Building vocab...')
    words = read_data(FILE_PATH)
    vocab = ['UNK']
    vocab.extend(w for w, _ in Counter(words).most_common(vocab_size - 1))
    utils.make_dir('processed')
    index = {}
    with open('processed/vocab.tsv', "w") as f:
        for i, v in enumerate(vocab):
            index[v] = i
            f.write(v.encode('utf-8') + "\n")
    return index

def convert_words_to_indicies(index):
    """ Replace each word in the dataset with its index in the dictionary """
    print('Converting words to indicies...')
    return [index[word] if word in index else 0 for word in read_data(FILE_PATH)]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    print('Generating sample...')
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target
    print('done')

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    print('Getting batch...')
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def process_data(vocab_size, batch_size, skip_window):
    index = build_vocab(vocab_size)
    indicies = convert_words_to_indicies(index)
    single_gen = generate_sample(indicies, skip_window)
    return get_batch(single_gen, batch_size)
