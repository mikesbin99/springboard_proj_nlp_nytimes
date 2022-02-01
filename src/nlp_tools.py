
import sys

import html
import re
from nltk import tokenize
import pandas as pd
import numpy as np

# Textacy
import textacy
import textacy.preprocessing as tprep # Preprocesing of accents/normalization
from textacy.preprocessing.resources import RE_URL

import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import constants 

from collections import Counter

import regex as rex

from spacy import displacy

displacy.render(doc, style='ent', jupyter=True)

stopwords = set(nltk.corpus.stopwords.words('english'))

# Stopwords to add or remove (good to do this based on domain)
include_stopwords = {'also'}
exclude_stopwords = {'against'}

stopwords |= include_stopwords
stopwords -= exclude_stopwords

#https://learning.oreilly.com/library/view/blueprints-for-text/9781492074076/ch01.html#idm46749295328472


# def createCorpus(text_list):
#     # Aggregate text into single block of text
#     # Used for making master list of all text and vocab
#     corpus_text = []
#     for text in text_list:
#         corpus_text.extend(text)
#     return corpus_text

def tokenizeText(text: str):
    """Should be first

    Args:
        text (str): [description]

    Returns:
        [list]: [description]
    """
    #tokenize / split words 
    tokenizer = RegexpTokenizer(r'\w+') # Consider (r'\w+') r'\w+|\$[\d\.]+|\S+'
    tokens = tokenizer.tokenize(text)
    return tokens

def lemmatizeTokenList(text_list: list):
    """Take tokens and reduce words to base root form

    Args:
        tokens (list): [description]

    Returns:
        [type]: [description]
    """
    lemmater = WordNetLemmatizer()
    lemms = [lemmater.lemmatize(token.lower(),pos='v') for token in text_list]
    return lemms

def expandContraction(text):
    # https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    # specific
    phrase = re.sub(r"won\'t", "will not", text)
    phrase = re.sub(r"can\'t", "can not", text)
    phrase = re.sub(r"don\'t", "do not", text)

    # # general
    # phrase = re.sub(r"n\'t", " not", text)
    # phrase = re.sub(r"\'re", " are", text)
    # phrase = re.sub(r"\'s", " is", text)
    # phrase = re.sub(r"\'d", " would", text)
    # phrase = re.sub(r"\'ll", " will", text)
    # phrase = re.sub(r"\'t", " not", text)
    # phrase = re.sub(r"\'ve", " have", text)
    # phrase = re.sub(r"\'m", " am", text)
    return phrase

def cleanStopWords(text_list: list):
    """Clean stop words from the list

    Args:
        text_list (list): [description]

    Returns:
        [type]: [description]
    """
    # stopwords
    clean_text_list = [word for word in text_list if word not in stopwords]
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

def calculate_tdidf(df, column='tokens', preprocess=None, min_df=2):
    # Weighted Frequency
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))

    # count tokens
    counter = Counter()
    df[column].map(update)

    # create DataFrame and compute idf
    idf_df = pd.DataFrame.from_dict(counter, orient='index', columns=['df'])
    idf_df = idf_df.query('df >= @min_df')
    idf_df['idf'] = np.log(len(df)/idf_df['df'])+0.1
    idf_df.index.name = 'token'
    return idf_df

# def oneHotBinarizer(vocabulary):
#     lb = MultiLabelBinarizer()
#     lb.fit([vocabulary])
#     lb.transform(words)

RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

# Finds impurity score
def impurity(text, min_len=15):
    """returns the ratio of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

# Removes noise with regular expressions
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

def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    # text = tprep.replace_emojis(text)
    return text

# Cant' be run on tokenized
#https://learning.oreilly.com/library/view/blueprints-for-text/9781492074076/ch01.html#idm46749295254344
def count_words(df, column='tokens', preprocess=None, min_freq=2):

    # process tokens and update counter
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)

    # create counter and run through all data
    counter = Counter()
    df[column].map(update)

    # transform counter into a DataFrame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'

    return freq_df.sort_values('freq', ascending=False)

def compute_idf(df, column='tokens', preprocess=None, min_df=2):

    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))

    # count tokens
    counter = Counter()
    df[column].map(update)

    # create DataFrame and compute idf
    idf_df = pd.DataFrame.from_dict(counter, orient='index', columns=['df'])
    idf_df = idf_df.query('df >= @min_df')
    idf_df['idf'] = np.log(len(df)/idf_df['df'])+0.1
    idf_df.index.name = 'token'
    return idf_df
    
def convertKeywordsToBag(df_column):
    df_column = df_column.apply(eval)
    print(df_column)
    # bag = {}
    # for _ in df_column:
    #     bag[_] += 1
    return bag



def makeWordDict(word_bag):
    # Take a bag
    pass

def getListfromKeyWordListStr(df_column_lists):
    print(df_column_lists)
    return df_column_lists.apply(eval)



def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.words(doc, **kwargs)]

def extract_noun_phrases(doc, preceding_pos=['NOUN'], sep='_'):
    patterns = []
    for pos in preceding_pos:
        patterns.append(f"POS:{pos} POS:NOUN:+")
    spans = textacy.extract.matches(doc, patterns=patterns)
    return [sep.join([t.lemma_ for t in s]) for s in spans]

def extract_entities(doc, include_types=None, sep='_'):

    ents = textacy.extract.entities(doc, 
             include_types=include_types, 
             exclude_types=None, 
             drop_determiners=True, 
             min_freq=1)
    
    return [sep.join([t.lemma_ for t in e])+'/'+e.label_ for e in ents]

def extract_nlp(doc):
    return {
    'lemmas'          : extract_lemmas(doc,
                                        exclude_pos = ['PART', 'PUNCT',
                                        'DET', 'PRON', 'SYM', 'SPACE'],
                                        filter_stops = False),
    'adjs_verbs'      : extract_lemmas(doc, include_pos = ['ADJ', 'VERB']),
    'nouns'           : extract_lemmas(doc, include_pos = ['NOUN', 'PROPN']),
    'noun_phrases'    : extract_noun_phrases(doc, ['NOUN']),
    'adj_noun_phrases': extract_noun_phrases(doc, ['ADJ']),
    'entities'        : extract_entities(doc, ['PERSON', 'ORG', 'GPE', 'LOC'])
    }