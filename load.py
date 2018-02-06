from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import config as cfg
import w2v

class Corpus:
    """
    This class stores words and provides some methods that handle the words
    """
    def __init__(self, name):
        self.name = name
        self.trimmed = False	
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS and PRD and PRC(Price)

    def addSentence(self, sentence):
        for word in w2v.tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    return s

def readLangs(corpus):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s.txt' % (corpus), encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    print(pairs)
    # Reverse pairs, make Lang instances
    corpus_QA = Corpus(corpus)

    return corpus_QA, pairs

def filterPair(p):
    return len(p[0].split(' ')) < cfg.MAX_LENGTH and \
        len(p[1].split(' ')) < cfg.MAX_LENGTH and \
        p[1].startswith(cfg.eng_prefixes)


def filterPairs(corpus_QA, pairs):
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
    
        for word in w2v.tokenize(input_sentence):
            if word not in corpus_QA.word2index:
                keep_input = False
                break

        for word in w2v.tokenize(output_sentence):
            if word not in corpus_QA.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


#return [pair for pair in pairs if filterPair(pair)]
def updateData(corpus_QA,corpus_file):
    lines = open('data/%s.txt' % (corpus_file), encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize

    sentences = [l for l in lines]
	
    for sentence in sentences:
        try:
            sentence = sentence.split('\t')[2]
            corpus_QA.addSentence(sentence)
        except:
            print(sentence)
    print("Updated words:")
    print(corpus_QA.name, corpus_QA.n_words)

def prepareData(corpus):
    corpus_QA, pairs = readLangs(corpus)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        corpus_QA.addSentence(pair[0])
        corpus_QA.addSentence(pair[1])
    print("Counted words:")
    print(corpus_QA.name, corpus_QA.n_words)
    pairs = filterPairs(corpus_QA, pairs)

    return corpus_QA, pairs
