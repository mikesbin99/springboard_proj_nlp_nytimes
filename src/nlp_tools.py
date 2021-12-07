
import sys

import html
import re
from nltk import tokenize
import pandas as pd

# Textacy
import textacy.preprocessing as tprep # Preprocesing of accents/normalization
from textacy.preprocessing.resources import RE_URL

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer 

from sklearn.preprocessing import MultiLabelBinarizer

import constants 

stopwords = set(nltk.corpus.stopwords.words('english'))

# Stopwords to add or remove (good to do this based on domain)
include_stopwords = {'also'}
exclude_stopwords = {'against'}

stopwords |= include_stopwords
stopwords -= exclude_stopwords

#https://learning.oreilly.com/library/view/blueprints-for-text/9781492074076/ch01.html#idm46749295328472


def createCorpus(text_list):
    # Aggregate text into single block of text
    # Used for making master list of all text and vocab
    corpus_text = []
    for text in text_list:
        corpus_text.extend(text)
    return corpus_text



def tokenizeText(text: str):
    #Yokenize / split words 
    tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+') # Consider (r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens


def lemmatizeTokenList(tokens: list):
    # Lemmatize
    # Create Root form
    lemmater = WordNetLemmatizer()
    lemms = [lemmater.lemmatize(token.lower(),pos='v') for token in tokens]
    return lemms


def cleanStopWords(words: list):
    # stopwords
    clean_text_list = [word for word in words if word not in stopwords]
    return clean_text_list

def countVectorizer(cleaned_words: list):
    # https://towardsdatascience.com/introduction-to-nlp-part-1-preprocessing-text-in-python-8f007d44ca96
    vect = CountVectorizer() #
    feature_matrix = vect.fit_transform(cleaned_words)

    feature_df = pd.DataFrame.sparse.from_spmatrix(feature_matrix)     # Return dataframe

    # Rename columns to actual mapping names
    col_mapping = {v:k for k,v in vect.vocabulary_.items()}

    for col in feature_df.columns:
        feature_df.rename(columns={col: col_mapping[col]}, inplace=True)
    return feature_df

def countVectorizerSimple(cleaned_words: list):
    # https://towardsdatascience.com/introduction-to-nlp-part-1-preprocessing-text-in-python-8f007d44ca96
    vect = CountVectorizer() #
    feature_matrix = vect.fit(cleaned_words)

    feature_df = pd.DataFrame.sparse.from_spmatrix(feature_matrix)     # Return dataframe

    # Rename columns to actual mapping names
    col_mapping = {v:k for k,v in vect.vocabulary_.items()}

    for col in feature_df.columns:
        feature_df.rename(columns={col: col_mapping[col]}, inplace=True)
    return feature_df


# REMOVE Suspicious Characters
#https://learning.oreilly.com/library/view/blueprints-for-text/9781492074076/ch04.html#ch04removenoiseregex

def tdidf():
    # Weighted Frequency
    transformer = TfidfTransformer()


def oneHotBinarizer(vocabulary):
    lb = MultiLabelBinarizer()
    lb.fit([vocabulary])
    lb.transform(words)



RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

def impurity(text, min_len=15):
    """returns the ratio of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)


def clean(text: str):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

