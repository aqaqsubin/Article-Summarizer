import os
import csv
import numpy as np
import tensorflow as tf
from glob import iglob


class Decoder:
    def __init__(self, options):
        self.model = options['model-type']
        self.inv_wv = options['inv_wv']
        self.corpus = options['corpus']
        self.sp = options['spm']
        
        if self.corpus :
            self.inv_glove_dict = {v: k for k, v in self.corpus.dictionary.items()}
    
    def __glove_decoding(self, idx_list):
        return [self.inv_glove_dict[idx] for idx in idx_list 
                if idx in self.inv_glove_dict]
    
    def __word2vec_decoding(self, idx_list):
        return [self.inv_wv[idx] for idx in idx_list 
                 if idx in self.inv_wv]
    
    def __sentencepiece_decoding(self, idx_list):
        return self.sp.decode_ids(idx_list)
    
    def decode(self, idx_list=None):
        
        if self.model is 'GloVe':
            decoding_vec_list = self.__glove_decoding(idx_list) 
        elif self.model is 'Word2Vec' :
            decoding_vec_list = self.__word2vec_decoding(idx_list)
        else:
            decoding_vec_list = self.__sentencepiece_decoding(idx_list)
        
        return decoding_vec_list
       
    
class IntegerDecoder:
    def __init__(self, options, filepaths=None):
        self.filepaths = filepaths
        
        self.model = options['model-type']
        self.inv_wv = options['inv_wv']
        self.corpus = options['corpus']
        
        self.inv_glove_dict = {v: k for k, v in self.corpus.dictionary.items()}
    
    def __get_token_matrix(self):
        token_list =[]
        
        for path in self.filepaths:
            f = open(path, 'r', newline="\n", encoding="utf-8")
            
            for [_, title, contents] in csv.reader(f):  
                tokens = [token for sent in contents for token in sent.split()]
                token_list.append(np.array(tokens))
                
            f.close()

        return token_list
    
    def __glove_decoding(self, idx_matrix):
        return list(map(lambda line: [self.inv_glove_dict[idx] 
                                      for idx in idx_list if idx in self.inv_glove_dict], idx_matrix))
    
    def __word2vec_decoding(self, idx_matrix):
        return list(map(lambda line: [self.inv_wv[idx] 
                                      for idx in idx_list if idx in self.inv_wv], idx_matrix))  
    
    def decode(self, idx_matrix=None):
        
        idx_matrix = self.__get_token_matrix() if idx_matrix is None else idx_matrix
        decoding_vec_list = self.__glove_decoding(idx_matrix) if self.model is 'GloVe' else self.__word2vec_decoding(idx_matrix)    
        
        return decoding_vec_list   
    