import codecs
from konlpy.tag import Twitter
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

tagger = Twitter()

def read_data(filename):
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

def tokenize(doc):
    return [':'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]


def make_w2v_model(data_path):
    train_data = read_data(data_path)
    train_docs = [row[2] for row in train_data]
    sentences = [tokenize(d) for d in train_docs]
    model = word2vec.Word2Vec(sentences,size=100, window=5, min_count=5, workers=4)
    word_vectors = model.wv

    return word_vectors

def load_w2v_model(save_path):
    return KeyedVectors.load(save_path) 
