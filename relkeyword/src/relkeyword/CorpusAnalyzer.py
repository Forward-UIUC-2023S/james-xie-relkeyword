import yake
from tqdm import tqdm
import spacy
import json
import numpy as np
from collections import defaultdict
import math

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader as api

from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
import json
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

import os, os.path
import pickle


class CorpusAnalyzer:
    def __init__(self, documents, corpus_name):
        # documents will be a python list of sentences/documents.
        if not os.path.exists("relkeyword_dir"):
            os.mkdir("relkeyword_dir")
        self.root_folder = "relkeyword_dir/" + corpus_name + "/"
        if not os.path.exists(self.root_folder):
            os.makedirs(self.root_folder)
        self.document_path = self.root_folder + "documents.txt"
        print ("Num Documents: " + str(len(documents)))
        with open(self.document_path, 'wb') as f:
            pickle.dump(documents, f)
        self.documents = documents
        
        self.corpus_name = corpus_name
        self.kw_extractor = yake.KeywordExtractor()
        self.freq_dict = dict()
        self.semantic_related = defaultdict(lambda: defaultdict(int))
        self.semantic_similar = defaultdict(lambda: defaultdict(int))
        self.candidate_keyphrases = []
        self.embedding_model = SentenceTransformer('whaleloops/phrase-bert')
        self.phrase_indexing = dict()
        self.idf = defaultdict(int)
        # self.sentences = []
        
        
        
        
    def generate_candidate_keyphrases(self, threshold = 25):
        print ("Generating Candidate KeyPhrases")
        for i in tqdm(range(len(self.documents))):
            doc = self.documents[i]
            curr_keyphrases = self.kw_extractor.extract_keywords(doc)
            self.increment_counts(curr_keyphrases)
        
        
        freq_dict2 = dict()
        for keyphrase in self.freq_dict.copy():
            
            if (self.freq_dict[keyphrase] >= threshold):
                # del self.freq_dict[keyphrase.lower()]
                freq_dict2[keyphrase.lower()] = self.freq_dict[keyphrase]
        self.freq_dict = freq_dict2.copy()
        self.candidate_keyphrases = list(self.freq_dict.keys())

        cand_path = self.root_folder + "candidate_keyphrases.txt"
        with open(cand_path, 'w') as f:
            f.write(str(self.candidate_keyphrases))
    
    def increment_counts(self, words):
        if words is None:
            return

        for pair in words:
            word, num = pair
            if word not in self.freq_dict:
                self.freq_dict[word] = 1
            else:
                self.freq_dict[word] += 1
                
                
    def generate_related_keywords(self, candidate_keyphrases = None):
        noun_occurrence = defaultdict(int)
        pair_cooccur = defaultdict(lambda: defaultdict(int))
        num_nouns = 0
        total_cooccur = defaultdict(int)
        print("Parsing Documents")
        sentences = []
        # phrase_indexing = dict()
        sentence_index = 0
        documents = []
        with open(self.document_path, 'rb') as f:
            documents = pickle.load(f)
        
        if (candidate_keyphrases is not None):
            self.candidate_keyphrases = candidate_keyphrases
        
        for i in tqdm(range(len(documents))):
            paper = documents[i]
            
            paper_sentences = sent_tokenize(paper)
            sentences.extend(paper_sentences)
            for sentence in paper_sentences:
                current_sentence = list()
                sentence_words = 0
                lowered = sentence.lower()
                words = word_tokenize(sentence)
                sentence_words = len(words)
                
                for phrase in self.candidate_keyphrases:
                    if (phrase in lowered):
                        current_sentence.append(phrase)
                        noun_occurrence[phrase] += 1
                        num_nouns += 1
                        sentence_words += 1
                        if (phrase not in self.phrase_indexing):
                            self.phrase_indexing[phrase] = list()
                        self.phrase_indexing[phrase].append(sentence_index)
           
                for phrase in current_sentence:
                    for other_phrase in current_sentence:
                        if (phrase is not other_phrase):
                            pair_cooccur[phrase][other_phrase] += 1
                    total_cooccur[phrase] += sentence_words
                    self.idf[phrase] += 1
                sentence_index += 1

        
        for target in noun_occurrence:
            for context in pair_cooccur[target]:
                p_c_given_t = pair_cooccur[target][context]/total_cooccur[target]
                p_c = noun_occurrence[context]/num_nouns
                p_t_given_c = pair_cooccur[context][target]/total_cooccur[context]
                self.semantic_related[target][context] = p_t_given_c * p_c_given_t * (math.log(p_c_given_t/p_c))

        
        semantic_related_path = self.root_folder + "semantic_related.json"
        with open(semantic_related_path, 'w') as convert_file:
            convert_file.write(json.dumps(self.semantic_related))
            
        copy1 = defaultdict(lambda: defaultdict(int))
        for key in self.semantic_related:
            for pair in self.semantic_related[key]:
                if (key is pair or pair in key):
                    continue
                copy1[key][pair] = self.semantic_related[key][pair]
        copy = defaultdict(int) 

        for key in self.semantic_related:
            copy[key] = sorted(copy1[key].items(), key = lambda x: x[1])[-20:]
            copy[key] = copy[key][::-1]
        
        sorted_semantic_related_path = self.root_folder + "sorted_semantic_related.json"
        with open(sorted_semantic_related_path, 'w') as convert_file:
            convert_file.write(json.dumps(copy))
            
            
        indexing = {k: list(v) for k, v in self.phrase_indexing.items()}
        indexing_path = self.root_folder + "phrase_indexing.json"
        with open(indexing_path, 'w') as convert_file:
            convert_file.write(json.dumps(indexing))
        idf_path = self.root_folder + "idf.json"
        with open(idf_path, 'w') as convert_file:
            convert_file.write(json.dumps(self.idf))
            
        sentence_path = self.root_folder + 'sentences'
        with open(sentence_path, 'wb') as convert_file:
            pickle.dump(sentences, convert_file)
        return self.semantic_related, self.phrase_indexing, self.idf
                    
    def generate_semantic_similar(self, candidate_keyphrases = None):
        
        if (candidate_keyphrases is not None):
            self.candidate_keyphrases = candidate_keyphrases
        phrase_embs = self.embedding_model.encode(self.candidate_keyphrases)
        for phrase, embedding in tqdm(zip(self.candidate_keyphrases, phrase_embs)):
    
            for other_phrase, other_embedding in zip(self.candidate_keyphrases, phrase_embs):
                if (phrase is not other_phrase):
                    # semantic_similar[phrase][other_phrase] = cos_sim( torch.tensor(embedding), torch.tensor(other_embedding))
                    self.semantic_similar[phrase][other_phrase] = str(dot(embedding, other_embedding)/(norm(embedding) * norm(other_embedding)))
        
        semantic_similar_path = self.root_folder + "semantic_similar.json"
        with open(semantic_similar_path, 'w') as convert_file:
            convert_file.write(json.dumps(self.semantic_similar))
        
        copy1 = defaultdict(lambda: defaultdict(int))
        for key in self.semantic_related:
            for pair in self.semantic_related[key]:
                if (key is pair or pair in key):
                    continue
                copy1[key][pair] = self.semantic_similar[key][pair]
        copy = defaultdict(int) 

        for key in self.semantic_similar:
            copy[key] = sorted(copy1[key].items(), key = lambda x: x[1])[-20:]
            copy[key] = copy[key][::-1]
        
        sorted_semantic_similar_path = self.root_folder + "sorted_semantic_similar.json"
        with open(sorted_semantic_similar_path, 'w') as convert_file:
            convert_file.write(json.dumps(copy))
        return self.semantic_similar
        
                
    
    
    