import spacy
from tqdm import tqdm
import json
import numpy as np
import networkx as nx
from collections import defaultdict
import re

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader as api

from parse_csv import parse_csv
# wv = api.load('word2vec-google-news-300')
# model = Word2Vec(sentences=common_texts, vector_size=3, window=5, min_count=1, workers=4, sg=1)
# model.save("word2vec.model")
nlp = spacy.load("en_core_web_sm")
model = gensim.models.Word2Vec.load("word2vecdoc.model")
# Arxiv category to filter by (in the below case, regex will match computer science category). See https://arxiv.org/category_taxonomy
# filter_categ_re = re.compile(r"physics.gen-ph")
# with open("general_count.txt", 'r') as general_counts:
#     general_word_count = json.load(general_counts)

# Extract unigram frequency counts from "unigram_freq.csv" which is extracted form Google Web Trillion Word Corpus
import pandas as pd
df = pd.read_csv('unigram_freq.csv', header=None, index_col=0, squeeze = True)
unigram_word_counts = df.to_dict()



def extract_candidate_keywords(papers, seed_words):
    candidate_keywords = set() # Set of all unique candidate keywords
    
    embedding_similarity = defaultdict(int) # Map from candidate keyword to cosine similarity with initial keywords
    total_words = [] # For word2vec conversion
    co_occurence_similarity = defaultdict(lambda: defaultdict(int)) # Map from candidate keyword to co-occurence similarity
    initial_word_frequency = defaultdict(int)
    N_d = 0
    word_frequencies = defaultdict(int)
    domain_relevance = defaultdict(int)
    doc_word_count = defaultdict(int)
    for i in tqdm(range(len(papers))):
        paper = papers[i]
        # paper = json.loads(paper)
        
        # paper = paper['abstract']
        doc = nlp(paper)
        current_paper_words = set()
        abstract = []
        initial_word_in_doc = dict()
        
        
        # Check if any of the initial words are in this document
        for initial_word in seed_words:
             if (paper.find(initial_word) != -1):
                initial_word_in_doc[initial_word] = True
                initial_word_frequency[initial_word] += 1
            
            
        # Iterate through all the tokens in the document and only add POS tagged Nouns and Adjectives
        for token in doc:
            if ("https" not in token.text):
                if (token.pos_ == "NOUN" or token.pos_ == "ADJ"):
                    if (token.text.lower() not in current_paper_words):
                        doc_word_count[token.text.lower()] += 1
                    candidate_keywords.add(token.text.lower())
                    current_paper_words.add(token.text.lower())
                    word_frequencies[token.text.lower()] += 1
                    
            abstract.append(token.text.lower())
            
        
        total_words.append(abstract)
        N_d += len(abstract)
        
        # Append one count for each co_occurence_similarity
        for word in current_paper_words:
            for initial_word in initial_word_in_doc:
                co_occurence_similarity[word][initial_word] += 1
        

    total_words.append(seed_words)
    # print (total_words)
    # model = gensim.models.Word2Vec(total_words, workers=8, vector_size = 500,
    #                                          window = 5, sg = 1, min_count=1)
    # model.save("word2vecdoc.model")
    
    co_occurence = defaultdict(int)
    for word in candidate_keywords:
        weight = 0
        co_occurence_sum = 0
        if word in model.wv.key_to_index:
            for initial_word in seed_words:
                weight += model.wv.similarity(word, initial_word)
                co_occurence_sum += co_occurence_similarity[word][initial_word]/doc_word_count[word]
        weight /= len(seed_words)
        embedding_similarity[word] = weight
        co_occurence[word] = co_occurence_sum/len(seed_words)
        
        
        if (word not in unigram_word_counts):
            domain_relevance[word] = 0
        else: 
            general_corpora_prob = unigram_word_counts[word]/1024908267229.0
            word_prob = word_frequencies[word]/N_d
            domain_relevance[word] = (word_prob/general_corpora_prob)
            
    total_sum = sum(domain_relevance.values())
    for word in domain_relevance:
        domain_relevance[word] /= total_sum
        
    total_sum = sum(co_occurence.values())
    for word in co_occurence:
        co_occurence[word] /= total_sum
        
    return candidate_keywords, embedding_similarity, co_occurence, domain_relevance, co_occurence_similarity, doc_word_count

def calculate_vertex_weights(candidate_keywords, embedding_similarity, co_occurence, domain_relevance):
    vertex_weights = defaultdict(int)
    for word in candidate_keywords:
        # vertex_weights[word] = (1/3) * (embedding_similarity[word] + co_occurence[word] + domain_relevance[word])
        vertex_weights[word] = embedding_similarity[word]
    return vertex_weights
        
        
def construct_graph(candidate_keywords, seed_words, co_occurence_similarity, doc_word_count):
    N = len(candidate_keywords)
    M = len(seed_words)
    indexed_array = np.array(list(candidate_keywords))
    # print (indexed_array)
    
    co_graph = np.zeros((N, M))
    for i in range(N):
        word = indexed_array[i]
        for j in range(M):
            initial_word = seed_words[j]
            if (co_occurence_similarity[word][initial_word] > 0):
                co_graph[i][j] = co_occurence_similarity[word][initial_word]
                
    max_edge = np.amax(co_graph)

    for i in range(N):
        for j in range(M):
            co_graph[i][j]/= max_edge    
    
        # index = np.where(indexed_array == 4)[0][0]
    return co_graph, indexed_array
        
def diffusion_algorithm(co_graph, vertex_weights, seed_words, indexed_array):
    
    initial_word_weights = dict()
    for j in range(len(co_graph[0])):
        initial_word = seed_words[j]
        initial_word_weight = 0
        for i in range(len(co_graph)):
            candidate_word = indexed_array[i]
            r_i = vertex_weights[candidate_word]
            k_CW = np.sum(co_graph[i])
            if (k_CW == 0):
                continue
            weight = co_graph[i][j]
            initial_word_weight += weight * (r_i/k_CW)
        initial_word_weights[initial_word] = initial_word_weight
    
    for i in range(len(co_graph)):
        candidate_word = indexed_array[i]
        candidate_word_weight = 0
        for j in range(len(co_graph[0])):
            initial_word = seed_words[j]
            r_i = initial_word_weights[initial_word]
            
            k_IW = np.sum(co_graph[:, j])
            if (k_IW == 0):
                continue
            weight = co_graph[i][j]
            candidate_word_weight += weight * (r_i/k_IW)
        vertex_weights[candidate_word] = candidate_word_weight
    return vertex_weights, initial_word_weights
            
                 
import math
        
    
def main(papers, seed_words):
    candidate_keywords, embedding_similarity, co_occurence, domain_relevance, co_occurence_similarity, doc_word_count = extract_candidate_keywords(papers, seed_words)
    with open('embedding_similarity.txt', 'w') as convert_file:
        convert_file.write(json.dumps(embedding_similarity))
        
    with open('domain_relevance.txt', 'w') as convert_file:
        convert_file.write(json.dumps(domain_relevance))
        
    with open('co_occurrence_similarity.txt', 'w') as convert_file:
        convert_file.write(json.dumps(co_occurence_similarity))
    
    with open('co_occurrence.txt', 'w') as convert_file:
        convert_file.write(json.dumps(co_occurence))
    vertex_weights = calculate_vertex_weights(candidate_keywords, embedding_similarity, co_occurence, domain_relevance)
    with open('vertex_weights.txt', 'w') as convert_file:
        convert_file.write(json.dumps(vertex_weights))
    print("CONSTRUCTING GRAPH")
    co_graph, indexed_array = construct_graph(candidate_keywords, seed_words, co_occurence_similarity, doc_word_count)
    final_vertex_weights, initial_word_weights = diffusion_algorithm(co_graph, vertex_weights, seed_words, indexed_array)
    # nonnandict = filter(lambda k: not math.isnan(k), vertex_weights)

    sorted_dict = sorted(final_vertex_weights.items(), key=lambda x: x[1])
    with open('results.txt', 'w') as convert_file:
        convert_file.write(json.dumps(sorted_dict))
    print (sorted_dict)



# Testing diffusion implementation
# co_graph = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [2, 4, 1, 0], [0, 1, 0, 1], [0, 0, 3, 2]])
# vertex_weights = defaultdict(int)
# vertex_weights["CW1"] = 1
# vertex_weights["CW3"] = 1
# seed_words = ["IK1", "IK2", "IK3", "IK4"]
# indexed_array = np.array(["CW1", "CW2", "CW3", "CW4", "CW5"])
# vertex_weights, iniital_word_weights = diffusion_algorithm(co_graph, vertex_weights, seed_words, indexed_array)
# print (vertex_weights)
# print (iniital_word_weights)
# testing = ['''Domain-specific keyword extraction is a vital task in the field of text mining. 
#            There are various research tasks, such as spam e-mail classification, abusive language detection, sentiment analysis, and emotion mining, where a set of domain-specific keywords (aka lexicon) is highly effective. 
#            Existing works for keyword extraction list all keywords rather than domain-specific keywords from a document corpus. Moreover, most of the existing approaches perform well on formal document corpuses but fail on noisy and informal user-generated content in online social media.
#            In this article, we present a hybrid approach by jointly modeling the local and global contextual semantics of words, utilizing the strength of distributional word representation and contrasting-domain corpus for domain-specific keyword extraction. Starting with a seed set of a few domain-specific keywords, we model the text corpus as a weighted word-graph. 
#            In this graph, the initial weight of a node (word) represents its semantic association with the target domain calculated as a linear combination of three semantic association metrics, and the weight of an edge connecting a pair of nodes represents the co-occurrence count of the respective words. Thereafter, a modified PageRank method is applied to the word-graph to identify the most relevant words for expanding the initial set of domain-specific keywords. We evaluate our method over both formal and informal text corpuses (comprising six datasets), and show that it performs significantly better in comparison to state-of-the-art methods. 
#            Furthermore, we generalize our approach to handle the language-agnostic case, and show that it outperforms existing language-agnostic approaches.''']
# seed_words = ["domain", "keywords", "extraction"]
# candidate_keyords, embedding_simililarity = extract_candidate_keywords(testing, seed_words)
# print (embedding_simililarity)
        
        
