# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class Vocab(object):
    def __init__(self, vocab_file=None):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.pad_sym = "<pad>"
        self.eos_sym = "<eos>"
        self.unk_sym = "<unk>"

        self.insert(self.pad_sym)
        self.insert(self.unk_sym)
        self.insert(self.eos_sym)

        if vocab_file is not None:
            self.load_vocab(vocab_file)

    def insert(self, token):
        if token not in self.word2id:
            index = len(self.word2id)
            self.word2id[token] = index
            self.id2word[index] = token

            self.word2count[token] = 0
        self.word2count[token] += 1

    def size(self):
        return len(self.word2id)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as reader:
            for token in reader:
                self.insert(token.strip())

    def get_token(self, id):
        if id in self.id2word:
            return self.id2word[id]
        return self.unk_sym

    def get_id(self, token):
        if token in self.word2id:
            return self.word2id[token]
        return self.word2id[self.unk_sym]

    def sort_vocab(self):
        sorted_word2count = sorted(
            self.word2count.items(), key=lambda x: - x[1])
        self.word2id, self.id2word = {}, {}
        self.insert(self.pad_sym)
        self.insert(self.unk_sym)
        self.insert(self.eos_sym)
        for word, _ in sorted_word2count:
            self.insert(word)

    def save_vocab(self, vocab_file, size=1e6):
        with open(vocab_file, 'w') as writer:
            for id in range(min(self.size(), int(size))):
                writer.write(self.id2word[id] + "\n")

    def to_id(self, tokens, append_eos=True):
        if not append_eos:
            return [self.get_id(token) for token in tokens]
        else:
            return [self.get_id(token) for token in tokens + [self.eos_sym]]

    def to_tokens(self, ids):
        return [self.get_token(id) for id in ids]

    def eos(self):
        return self.get_id(self.eos_sym)

    def pad(self):
        return self.get_id(self.pad_sym)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vocabulary Preparison')
    parser.add_argument('--size', type=int, default=1e6, help='maximum vocabulary size')
    parser.add_argument('input', type=str, help='the input file path')
    parser.add_argument('output', type=str, help='the output file name')

    args = parser.parse_args()

    vocab = Vocab()
    with open(args.input, 'r') as reader:
        for line in reader:
            for token in line.strip().split():
                vocab.insert(token)

    vocab.sort_vocab()
    vocab.save_vocab(args.output, args.size)

    print("Loading {} tokens from {}".format(vocab.size(), args.input))
