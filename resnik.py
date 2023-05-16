import spacy
from tqdm import tqdm
import json
import numpy as np
import networkx as nx
from collections import defaultdict
import re
import math
nlp = spacy.load("en_core_web_sm")


def extract_candidate_keywords(papers):
    noun_occurrence = defaultdict(int)
    pair_cooccur = defaultdict(lambda: defaultdict(int))
    num_nouns = 0
    total_cooccur = defaultdict(int)
    for i in tqdm(range(len(papers))):
    # for i in range(1):
        paper = papers[i]
        doc = nlp(paper)
        # paper_words = 0
        # current_paper = list()
        for sentence in doc.sents:
            current_sentence = list()
            sentence_words = 0
            for token in sentence:
                if (token.pos_ == "NOUN"):
                    word = token.text.lower()
                    noun_occurrence[word] += 1
                    num_nouns += 1
                    current_sentence.append(word)
                    sentence_words += 1
            for word in current_sentence:
                for other_word in current_sentence:
                    if (word != other_word):
                        pair_cooccur[word][other_word] += 1
                total_cooccur[word] += sentence_words
                    
        # for token in doc:
        #     if (token.pos_ == "NOUN"):
        #         word = token.text.lower()
        #         noun_occurrence[word] += 1
        #         num_nouns += 1
        #         current_paper.append(word)
        #         paper_words += 1
    
        # for word in current_paper:
        #     for other_word in current_paper:
        #         if (word != other_word):
        #             pair_cooccur[word][other_word] += 1
        #     total_cooccur[word] += paper_words
                    
    pair_scores = defaultdict(lambda: defaultdict(int))
    for target in noun_occurrence:
        for context in pair_cooccur[target]:
            p_c_given_t = pair_cooccur[target][context]/total_cooccur[target]
            # if (p_c_given_t < 0.007):
            #     continue
            p_c = noun_occurrence[context]/num_nouns
            p_t_given_c = pair_cooccur[context][target]/total_cooccur[context]
            pair_scores[target][context] = p_t_given_c * p_c_given_t * (math.log(p_c_given_t/p_c))

    return pair_scores
        
def resnik(papers):
    pair_scores = extract_candidate_keywords(papers)
    # print (pair_scores)
    with open('output_files/sentence_pair_scores.txt', 'w') as convert_file:
        convert_file.write(json.dumps(pair_scores))
    