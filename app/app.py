from flask import Flask, abort, jsonify, request, render_template
import joblib
import json

from this import s

import numpy as np
import pandas as pd

import os
import re
import nltk


def get_all_query(title, author, text):
    total = title + author + text
    total = [total]
    return total


def remove_punctuation_stopwords_lemma(sentence, stop_words=None):
    filter_sentence = ''
    lemmatizer = nltk.WordNetLemmatizer()
    sentence = re.sub(r'[^\w\s]', '', s)
    words = nltk.word_tokenize(sentence)  # tokenization
    words = [w for w in words if not w in stop_words]
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
    return filter_sentence


pipeline = joblib.load('./pipeline.sav')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def get_delay():
    result = request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    print(query_text)
    query = get_all_query(query_title, query_author, query_text)
    query = remove_punctuation_stopwords_lemma(query)
    user_input = {'query': query}
    pred = pipeline.predict(query)
    print(pred)
    dic = {1: 'Real', 0: 'Fake'}
    return f'<html><body><h1>{dic[pred[0]]}</h1> <form action="/"> <button type="submit">back </button> ' \
           f'</form></body></html> '


if __name__ == '__main__':
    app.run(port=8080, debug=True)
