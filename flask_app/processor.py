import pandas as pd
import numpy as np
import json
from os import listdir
import os
import re
import pickle
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, pos_tag



##This file is used by the flask app to unpickle the text processing unit
## A refactor of this code would remove the need for duplicating this code

PARTSOFSPEECH = ['WRB', 'VBZ', 'DT', 'NN', '.', 'NNP', ',', 'PRP', 'VBD', 'TO',
       'NNS', 'IN', 'VBG', 'CC', 'RP', 'RB', 'MD', 'VB', 'VBP', 'JJ', ':',
       'CD', 'JJR', 'WDT', 'PRP$', 'VBN', 'POS', 'WP', 'JJS', '$',
       '-NONE-', 'NNPS', 'EX']

def getPOS(x):
## Function used to get parts of speech features
## Input: tokenized email list
## Output: Counts of parts of speech

    parts = [i[1] for i in pos_tag(x)]

    output = []
    for part in PARTSOFSPEECH:
        output.append(np.sum(np.array(parts) == part))
    return output / (np.linalg.norm(output, ord=2)+1)

class Processor(object):

        
    def __init__(self):
        self.myVect = None
        PARTSOFSPEECH = ['WRB', 'VBZ', 'DT', 'NN', '.', 'NNP', ',', 'PRP', 'VBD', 'TO',
       'NNS', 'IN', 'VBG', 'CC', 'RP', 'RB', 'MD', 'VB', 'VBP', 'JJ', ':',
       'CD', 'JJR', 'WDT', 'PRP$', 'VBN', 'POS', 'WP', 'JJS', '$',
       '-NONE-', 'NNPS', 'EX']
        pass
    
    def fit_transform(self, X, flag=True, verbose=True):
        
        if verbose:
            print 'Welcome to the fit_transform for the data'
        
        X = X.reset_index(drop=True)
        
        if verbose:
            print 'Base tfidf start:'
        ## TFIDF from sklearn
        if flag:
            self.myVect = TfidfVectorizer()
            self.myVect.fit(X)
            if verbose:
                print 'training successful'
            
        output = pd.DataFrame(self.myVect.transform(X).toarray())
        if verbose:
            print 'transform successful, output = ', output.shape

        output.columns = self.myVect.vocabulary_
        # Testing purposes
        output = pd.DataFrame([1]*X.shape[0])
        
        ## tokenize words
        if verbose:
            print 'Tokenizing text'
            
        tokens = X.map(word_tokenize)
        
        if verbose:
            print 'Tokenizing successful:', tokens.shape
        ## Count of words as feature
        output['word_count'] = tokens.map(len)
        
        
        ## Part of speech as feature (see getPOS for more detail)
        if verbose:
            print 'part of speech tagging'
        p = Pool(7)
        pos = pd.DataFrame(p.map(getPOS, tokens), columns=PARTSOFSPEECH)
        if verbose:
            print 'Tagging successful:', pos.shape
        output = pd.concat([output, pos], axis=1)
        
        
        return output
    
    def fit(self, *args):
        self.fit_transform(*args)
    
    def transform(self, X):
        output = self.fit_transform(X, flag=False)
        return output