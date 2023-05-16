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

from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
nlp = spacy.load("en_core_web_sm")
model = gensim.models.Word2Vec.load("word2vecdoc.model")
print ("FETCHING CANDIDATE WORDS")
with open("yake_output.json") as json_file:
    data = json.load(json_file)
    candidate_words = []
    for i in data:
        if (i[1] >= 5):
            candidate_words.append(i[0].lower())
print (len(candidate_words))

        
    

def extract_candidate_keywords(papers):
    noun_occurrence = defaultdict(int)
    pair_cooccur = defaultdict(lambda: defaultdict(int))
    num_nouns = 0
    total_cooccur = defaultdict(int)
    print("Parsing Documents")
    sentences = []
    phrase_indexing = dict()
    sentence_index = 0
    for i in tqdm(range(len(papers))):
    
    # for i in range(100):
        paper = papers[i]
        # doc = nlp(paper)
        paper_sentences = sent_tokenize(paper)
        sentences.extend(paper_sentences)
        # print (paper_sentences)
        for sentence in paper_sentences:
            # actual_sentence = list()
            current_sentence = list()
            sentence_words = 0
            # print(type(sentence))
            # sentences.append(sentence)
            # words = word_tokenize(sentence)
            lowered = sentence.lower()
            for phrase in candidate_words:
                if (phrase in lowered):
                    current_sentence.append(phrase)
                    noun_occurrence[phrase] += 1
                    num_nouns += 1
                    sentence_words += 1
                    if (phrase not in phrase_indexing):
                        phrase_indexing[phrase] = list()
                    phrase_indexing[phrase].append(sentence_index)
            for phrase in current_sentence:
                for other_phrase in current_sentence:
                    if (phrase is not other_phrase):
                        pair_cooccur[phrase][other_phrase] += 1
                total_cooccur[phrase] += sentence_words
            sentence_index += 1
                    
                    
            
            
            # for token in sentence:
            #     if (token.pos_ == "NOUN" or token.pos_ == "ADJ"):
            #         word = token.text.lower()
            #         noun_occurrence[word] += 1
            #         num_nouns += 1
            #         current_sentence.append(word)
            #         sentence_words += 1
            #         if (word not in word_indexing):
            #             word_indexing[word] = list()
            #         word_indexing[word].append(sentence_index)
            # for word in current_sentence:
            #     for other_word in current_sentence:
            #         if (word != other_word):
            #             pair_cooccur[word][other_word] += 1
            #     total_cooccur[word] += sentence_words
            # sentence_index += 1
                    
    semantic_related = defaultdict(lambda: defaultdict(int))
    semantic_similar = defaultdict(lambda: defaultdict(int))
    important_words = defaultdict()
    
    # for word in noun_occurrence:
    #     if (noun_occurrence[word] >= 8):
    #         important_words[word] = noun_occurrence[word]
    print ("Calculating Scores")
    for target in noun_occurrence:
        # if target not in important_words:
        #     continue
        for context in pair_cooccur[target]:
            # if context not in important_words:
            #     continue
            p_c_given_t = pair_cooccur[target][context]/total_cooccur[target]
            # if (p_c_given_t < 0.007):
            #     continue
            p_c = noun_occurrence[context]/num_nouns
            p_t_given_c = pair_cooccur[context][target]/total_cooccur[context]
            semantic_related[target][context] = p_t_given_c * p_c_given_t * (math.log(p_c_given_t/p_c))
            if target in model.wv.key_to_index and context in model.wv.key_to_index:
                semantic_similar[target][context] = str(model.wv.similarity(target, context))

    return semantic_related, semantic_similar, sentences, phrase_indexing
        
def extract_related_keywords(papers):
    
    semantic_related, semantic_similar, sentences,word_indexing = extract_candidate_keywords(papers)
    # print (pair_scores)
    with open('test3/semantic_related.json', 'w') as convert_file:
        convert_file.write(json.dumps(semantic_related))
        
    with open('test3/semantic_similar.json', 'w') as convert_file:
        convert_file.write(json.dumps(semantic_similar))
        
    
    with open('test3/sentences', 'wb') as convert_file:
        pickle.dump(sentences, convert_file)
    
    json_data = {k: list(v) for k, v in word_indexing.items()}

    with open('test3/word_indexing.json', 'w') as convert_file:
        convert_file.write(json.dumps(json_data))