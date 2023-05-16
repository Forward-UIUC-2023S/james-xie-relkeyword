#!/usr/bin/env python3
import re
import json
import pickle
import yake
import spacy
from collections import defaultdict

from tqdm import tqdm  
from parse_csv import parse_csv
from resnik import resnik

from tf_idf import tf_idf_main
arxiv_file = f"/Users/jamesxie/Desktop/FWD Data Lab/arxiv-metadata-oai-snapshot.json"
output_file = './yake_output.json'


# Minimum frequency needed among papers for a keyword to be a candidate
filter_thresh = 0

# Arxiv category to filter by (in the below case, regex will match computer science category). See https://arxiv.org/category_taxonomy
filter_categ_re = re.compile(r"physics.gen-ph")

# The number of top keywords to extract per paper when computing global frequency counts
num_kwds_per_paper = 10


def get_paper_data():
    with open(arxiv_file, 'r') as f:
        
        for line in f:
            yield line


# Implementation of graph to keep track of co-occurences of papers
freq_dict = {}

def increment_counts(words):
    if words is None:
        return

    for pair in words:
        word, num = pair
        if word not in freq_dict:
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1



# Parse data from papers (keyword search)
# embedding_distributor = launch.load_local_embedding_distributor()
# pos_tagger = launch.load_local_corenlp_pos_tagger()

papers = parse_csv()


max_papers = 500
max_papers = None
p_i = 0
tf_idf_main(papers)
    
# tf_idf_main(papers)