import load
import random
import re

random_ratio = 10 # 1 to 100
shuffle_ratio = 10
repetition = 100

train_path = "search3"
dict_path = "dict"

def make_noise_list(train_path,dict_path):
    noise_words = []
    train_corpus, _ = load.prepareData(train_path)
    dict_corpus = load.Corpus(dict_path)
    load.updateData(dict_corpus,dict_path)
	
    for i in range(dict_corpus.n_words):
        word = dict_corpus.index2word[i]
        try:
            word,morpheme = word.split(":")
            if morpheme == 'Noun':
                noise_words.append(word)
        except:
            pass

    return noise_words
#차후에 숫자를 문자로 바꾸어 처리하는 모듈을 삽입할 것!
def exchange_number_in_sentence(sentence):
    r = str(random.randrange(1000))
    numbers = re.findall(r'\d+', sentence)

    for n in numbers:
        sentence = sentence.replace(n,r)
	
    return sentence

def make_noise_sentences(noise_words,train_sentences):
    noise_sentences = []
    for sentence,answer in train_sentences:
        noise_sentence = []
        sentence = sentence.split(' ')
        for word in sentence:
            if random.randrange(100) % random_ratio == 1:
                noise_sentence.append(random.choice(noise_words))
            noise_sentence.append(word)
            
        if random.randrange(100) % random_ratio == 1:
            noise_sentence.append(random.choice(noise_words))
#noise_sentence = ' '.join(noise_sentence)            
        noise_sentences.append((noise_sentence,answer))

    return noise_sentences


def shuffle_words_in_sentence(noise_sentences):
    augmented_sentences = []
    for sentence, answer in noise_sentences:
        if random.randrange(100) % shuffle_ratio == 101:
            random.shuffle(sentence)
        sentence = ' '.join(sentence)
        augmented_sentences.append((sentence,answer))

    return augmented_sentences


def make_sentence_list(train_path):
    train_sentence = []

    with open("data/"+train_path+".txt","r") as fp:
        lines = fp.readlines()
	
    for line in lines:
        line = exchange_number_in_sentence(line)
        sentence,answer = line.split('	')
        train_sentence.append((sentence,answer))

    return train_sentence


def main():
    train_sentences = make_sentence_list(train_path)
    noise_words = make_noise_list(train_path,dict_path)
    noise_sentences = make_noise_sentences(noise_words,train_sentences)

    with open("augmented_train.txt","w") as fp:
        for i in range(repetition):
            noise_sentences = make_noise_sentences(noise_words,train_sentences)
            augmented_sentences = shuffle_words_in_sentence(noise_sentences)
            for q,a in augmented_sentences:
                fp.write("%s	%s\n" % (q,a.strip()))


if __name__=='__main__':
   main()
