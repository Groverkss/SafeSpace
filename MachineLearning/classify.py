from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

import matplotlib.pyplot as plt

sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one?',
    'Is this the first document? I am expecting some more???',
]

y = [1, 1, 0, 0]

classify = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression()),
])


def fit_model(split_data=False):
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.33, random_state=42
        )
        classify.fit(X_train, y_train)
        # y_pred = classify.predict(X_test)
        # print(confusion_matrix(y_test, y_pred))
        plot_confusion_matrix(classify, X_test, y_test)
        plt.show()
    else:
        classify.fit(sentences, y)

def predict(input_text):
    return classify.predict([input_text])

if __name__ == "__main__":
    fit_model(split_data=True)
    print(predict("Hello world"))