import random

import pickle
import numpy as np

from nltk.corpus import gutenberg
from nltk.corpus import webtext

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

import matplotlib.pyplot as plt

def make_data():
    global sentences
    global y
    nsfw_sentences = open("./NSFWCorpus/nsfw.txt").readlines()
    y1 = len(nsfw_sentences)
    general_sentences = open("./GeneralCorpus/general.txt").readlines()    
    y2 = len(general_sentences)
    nsfw_sentences = nsfw_sentences[:y2]
    y1 = len(nsfw_sentences)
    # print(y1)
    # print(y2)

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
        ('clf', LogisticRegression(max_iter=1000)),
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

def predict(input_text):
    classify = pickle.load(open('./MachineLearning/classifier.model', 'rb'))
    return classify.predict([input_text])
