import nltk
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenizes the input sentence into individual words.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stems the word to its root form.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Converts the tokenized sentence into a bag of words (vector form).
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = [0] * len(words)
    for idx, word in enumerate(words):
        if word in sentence_words:
            bag[idx] = 1
    return bag
