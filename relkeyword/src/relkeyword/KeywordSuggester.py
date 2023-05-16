import json
import os, os.path
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
class KeywordSuggester:
    def __init__(self, root_folder):
        if not os.path.exists(root_folder):
            print ("Failed to find directory. Check to make sure provided directory is correct.")
            return 
        semantic_related_path = root_folder + "sorted_semantic_related.json"
        semantic_similar_path = root_folder + "sorted_semantic_similar.json"
        with open(semantic_related_path, 'r') as f:
            semantic_related = f.read()
        
        with open(semantic_similar_path, 'r') as f:
            semantic_similar = f.read()
            
        semantic_similar = json.loads(semantic_similar)
        semantic_related = json.loads(semantic_related)
        
        phrase_indexing_path = root_folder + "phrase_indexing.json"
        with open(phrase_indexing_path) as f:
            word_indexing = f.read()
            
        word_indexing = json.loads(word_indexing)

        sentences_path = root_folder + "sentences"
        with open(sentences_path, 'rb') as f:
            self.sentences = pickle.load(f)
            
        idf_path = root_folder + "idf.json"
        with open(idf_path) as f:
            self.idf = f.read()
        self.idf = json.loads(self.idf)
            
        self.semantic_related = semantic_related
        self.semantic_similar = semantic_similar
        self.word_indexing = word_indexing
        
    def retrieve_related_words(self, keyword):
        
        if (keyword not in self.semantic_related or keyword not in self.semantic_similar):
            print ("Word not found in semantic recommendations")
            print ("Most likely word was not deemed a keyword by sufficient documents")
            return 
        
        similar_words = self.semantic_similar[keyword]
        related_words = self.semantic_related[keyword]

        new_related_words = []
        for i in range(len(related_words)):
            next_word = related_words[i]
            no_overlap = True
            for other_word in similar_words:
                if (next_word[0] == other_word[0]):
                    no_overlap = False
            if(no_overlap):
                new_related_words.append(next_word)

        if (len(new_related_words) >= 10):
            new_related_words = new_related_words[:10]
            
        data = {
            'word': keyword,
            'semantic_similar': similar_words,
            'semantic_related': new_related_words, 
        }

        return data
    
    def retrieve_sentence(self, keyword, related_word):
        print ("getting sentence")
        
        related_sentences = self.word_indexing[keyword]
        curr_sentences = self.word_indexing[related_word]
        if (len(related_sentences) > len(curr_sentences)):
            inter = set(related_sentences).intersection(curr_sentences)
        else: 
            inter = set(curr_sentences).intersection(related_sentences) 
        inter = list(inter)
        max_score = 0
        max_index = 0
        good_sentence = []
        idf_scores = []
        
        window_scores = []
        window_sentences = []
        idf_sentences = []
        for i in range(len(inter)):

            index = inter[i]
            sentence = self.sentences[index]
            if (len(sentence.split()) > 30):
                continue
            score = 0
            
            tf_one = sentence.count(keyword)
            tf_two = sentence.count(related_word)
            words = word_tokenize(sentence)
            
            # doc = nlp(sentence)
            # subj = False
            # for token in doc:
            #     if (token.dep_ == "nsubj"):
            #         token = token.text.lower()
            #         if (token == keyword or token == related_word):
            #             good_sentence.append(i)
            #             subj = True
                        
            tf_idf_score = 0
            if (self.idf[related_word] == 0 and self.idf[keyword] == 0):
                tf_idf_score = 0
            elif (self.idf[related_word] == 0):
                tf_idf_score = tf_one * 1/self.idf[keyword]
            elif (self.idf[keyword] == 0):
                tf_idf_score = tf_two * 1/self.idf[related_word]
            else:
                tf_idf_score = (tf_one * 1/self.idf[keyword]) + (tf_two * 1/self.idf[related_word])
            if (tf_idf_score > max_score):
                max_index = i
                max_score = score
            
            idf_scores.append(tf_idf_score)
            
            words = word_tokenize(sentence)
            if (keyword not in words or related_word not in words):
                continue
            index_one = words.index(keyword)
            index_two = words.index(related_word)
            if (index_one != -1 and index_two != -1 and (abs(index_two - index_one) < 5)):
                window_scores.append(tf_idf_score)
                window_sentences.append(i)
        
        if (len(window_scores) != 0):
            # idf_scores2 = []
            # for sentence in window_sentences:
            #     index12 = inter.index(sentence)
            #     idf_scores2.append(idf_scores[index12])
            max_index = np.argmax(window_scores)
            return {'sentence': self.sentences[inter[window_sentences[max_index]]]}
        # if (len(window_scores) != 0):
        #     min_index = np.argmin(window_scores)
        #     return {'sentence': self.sentences[inter[window_sentences[min_index]]]}
        # if (len(good_sentence) != 0):
        #     max_index = np.argmax(idf_scores)
        #     # return JsonResponse({'sentence': sentences[inter[good_sentence[0]]]})
        #     return {'sentence': self.sentences[inter[good_sentence[max_index]]]}
        
        sentence = self.sentences[inter[max_index]]
        data = {'sentence': sentence}
        return data
        