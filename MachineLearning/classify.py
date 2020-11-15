import random

import sys
sys.path.append("../")

import pickle
import numpy as np

from nltk.corpus import gutenberg
from nltk.corpus import webtext

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

import matplotlib.pyplot as plt

def make_data():
    global sentences
    global y
    nsfw_sentences = []
    with open("../NSFWCorpus/nsfw.txt") as f:
        nsfw_sentences = f.readlines()
    y1 = len(nsfw_sentences)

    general_sentences = []
    for corpus in [gutenberg, webtext]:
        for fileid in corpus.fileids():
            sentences = corpus.raw(fileid)
            
            if len(sentences) > 64050:
                sentences = sentences[:64050]

            for sentence in sentences:
                general_sentences.append(sentence)
    y2 = len(general_sentences)

    sentences = [sentence for sentence in nsfw_sentences]
    sentences += [sentence for sentence in general_sentences]
    y = [0]*y1
    y += [1]*y2

    data = [(sentence, label) for sentence, label in zip(sentences, y)]
    random.shuffle(data)

    sentences = []
    y = []
    for sentence, y_temp in data:
        sentences.append(sentence)
        y.append(y_temp)
        # sentences, y = list(zip(*data))

def make_model():
    global classify
    classify = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', svm.SVC()),
    ])

def fit_model(split_data=True):
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.20, random_state=42
        )
        classify.fit(X_train, y_train)
        # y_pred = classify.predict(X_test)
        # print(confusion_matrix(y_test, y_pred))
        plot_confusion_matrix(classify, X_test, y_test)
        plt.show()
    else:
        classify.fit(sentences, y)

def save_model():
    pickle.dump(classify, open('classifier.model', 'wb'))

def load_model():
    global classify
    classify = pickle.load(open('classifier.model', 'rb'))

def predict(input_text):
    return classify.predict([input_text])

if __name__ == "__main__":
    make_data()
    print("Loaded Data")
    make_model()
    print("Loaded Model")
    fit_model()
    print("Fitted Model")
    save_model()
    print("Saved Model")

    # load_model()
    # print(predict("sex"))