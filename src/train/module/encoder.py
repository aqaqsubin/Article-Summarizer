import os
import csv
import numpy as np
import tensorflow as tf
from glob import iglob

class IntegerEncoder:
    def __init__(self, filepaths, options):
        self.filepaths = filepaths
        
        self.model = options['model-type']
        self.inv_wv = options['inv_wv']
        self.corpus = options['corpus']
        self.sp = options['spm']
    
    def __get_token_matrix(self):
        token_list =[]
        
        for path in self.filepaths:
            f = open(path, 'r', newline="\n", encoding="utf-8")
            
            for [_, title, contents] in csv.reader(f):
                content = contents.split("\t")
                vec = [token for sent in content for token in sent.split()]

                token_list.append(np.array(vec))
                
            f.close()

        return token_list

    def __get_line_list(self):
        line_list =[]
        
        for path in self.filepaths:
            f = open(path, 'r', newline="\n", encoding="utf-8")
            
            for [_, title, contents] in csv.reader(f):
                content = contents.split("\t")
                line_list.append(' '.join(content))
                
            f.close()

        return line_list
    
    def __glove_encoding(self, token_list):
        return list(map(lambda line: [self.corpus.dictionary[token] for token in line 
                                      if token in self.corpus.dictionary], token_list))
        
    def __sentencepiece_encoding(self, token_list):
        return list(map(lambda line: self.sp.EncodeAsIds(line), token_list))
    
    def __sentencepiece_encoding_line(self, line):
        return self.sp.EncodeAsIds(line)
    
    def __word2vec_encoding(self, token_list):
        return list(map(lambda line: [self.inv_wv[token] for token in line
                                     if token in self.inv_wv], token_list))  
    
    def encoder(self):

        token_list = self.__get_token_matrix()
        if self.model is 'GloVe':
            encoding_vec_list = self.__glove_encoding(token_list) 
        elif self.model is 'Word2Vec' :
            encoding_vec_list = self.__word2vec_encoding(token_list)
        else:
            encoding_vec_list = self.__sentencepiece_encoding(self.__get_line_list())
        
        return encoding_vec_list
    
    def line_encoder(self, line=None):
        if not line :
            return
        
        return self.__sentencepiece_encoding_line(line)
    
