import spacy
from tqdm import tqdm
import json
import numpy as np
import networkx as nx
from collections import defaultdict
import re
import math

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader as api

import pickle
nlp = spacy.load("en_core_web_sm")

def tf_idf(papers):
    sentence_index = 0
    term_frequency = defaultdict(lambda: defaultdict(int))
    inverse_doc_frequency = defaultdict(int)
    sentence_to_doc = defaultdict(int)
    total_num_papers = len(papers)
    for i in tqdm(range(total_num_papers)):
        
    # for i in range(100):
        paper = papers[i]
        doc = nlp(paper)
        num_words = 0
        document = set()
        
        for sentence in doc.sents:
            # current_sentence = set()
            # sentence_words = 0
            sentence_to_doc[sentence_index] = i
            # for token in sentence:
            #     word = token.text.lower()
            #     term_frequency[i][word] += 1
            #     num_words += 1
            #     # current_sentence.add(word)
            #     # sentence_words += 1
            #     document.add(word)
            sentence_index += 1
            
    #     for word in document:
    #         term_frequency[i][word] /= num_words
    #         inverse_doc_frequency[word] += 1
        
            
    # for k,v in inverse_doc_frequency.items():               # will become d.items() in py3k
    #     inverse_doc_frequency[k] = v / total_num_papers
  

    return term_frequency, inverse_doc_frequency, sentence_to_doc
        
def tf_idf_main(papers):
    term_frequency, inverse_doc_frequency, sentence_to_doc = tf_idf(papers)
    # print (pair_scores)
        
    with open('data/test3/sentence_to_doc.json', 'w') as convert_file:
        convert_file.write(json.dumps(sentence_to_doc))
    # with open('test2/sentences', 'wb') as convert_file:
    #     pickle.dump(sentences, convert_file)
    
    # json_data = {k: list(v) for k, v in word_indexing.items()}

    # with open('test2/word_indexing.json', 'w') as convert_file:
    #     convert_file.write(json.dumps(json_data))
    
# tf_idf_main()