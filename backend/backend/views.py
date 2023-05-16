
from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
import json
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####

from collections import defaultdict
import pickle
import spacy
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
nlp = spacy.load("en_core_web_sm")

with open('../data/test3/semantic_similar2.json') as f:
    semantic_similar = f.read()
    
semantic_similar = json.loads(semantic_similar)

with open('../data/test3/sorted_semantic_related2.json') as f:
    semantic_related = f.read()
    
semantic_related = json.loads(semantic_related)
with open('../data/test3/word_indexing.json') as f:
    word_indexing = f.read()
    
word_indexing = json.loads(word_indexing)

with open('../data/test3/sentences', 'rb') as f:
    sentences = pickle.load(f)
# import wikipedia


with open('../data/test3/tf.json') as f:
    tf = f.read()
tf = json.loads(tf)

with open('../data/idf.json') as f:
    idf = f.read()
idf = json.loads(idf)

with open('../data/test3/sentence_to_doc.json') as f:
    sentence_to_doc = f.read()
sentence_to_doc = json.loads(sentence_to_doc)


def index(request):
    return HttpResponse("Hello, world. You're at the wiki index.")


def get_related_words(request):
    
    word = request.GET.get('topic', None)
    print('word:', word)
    if (word not in semantic_related or word not in semantic_similar):
        print ("DID NOT FIND WORD")
        return JsonResponse({'word': "-1"})
   
    similar_words = semantic_similar[word]
    related_words = semantic_related[word]

    new_related_words = []
    for i in range(len(related_words)):
        next_word = related_words[i]
        no_overlap = True
        for other_word in similar_words:
            print (next_word)
            print (other_word)
            if (next_word[0] == other_word[0]):
                no_overlap = False
        if(no_overlap):
            new_related_words.append(next_word)
                # related_words.pop(i)
        # if next_word in similar_words:
        #     print(next_word)
        #     del related_words[next_word]
    
    if (len(new_related_words) >= 10):
        new_related_words = new_related_words[:10]
    data = {
        'word': word,
        'semantic_similar': similar_words,
        'semantic_related': new_related_words, 
    }

    print('json-data to be sent: ', data)

    return JsonResponse(data)

def get_sentence(request):
    print ("getting sentence")
    word = request.GET.get('word')
    related_word = request.GET.get('related_word')
    
    related_sentences = word_indexing[word]
    curr_sentences = word_indexing[related_word]
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
    for i in range(len(inter)):

        index = inter[i]
        sentence = sentences[index]
        if (len(sentence.split()) > 30):
            continue
        score = 0
        
        tf_one = sentence.count(word)
        tf_two = sentence.count(related_word)
        words = word_tokenize(sentence)
        
        doc = nlp(sentence)
        subj = False
        for token in doc:
            if (token.dep_ == "nsubj"):
                token = token.text.lower()
                if (token == word or token == related_word):
                    good_sentence.append(i)
                    subj = True
        #     if (word not in term_frequencies):
        #         continue

        #     tf_idf_score = term_frequencies[word] * 1/idf[word]
        #     sum += tf_idf_score
        tf_idf_score = 0
        if (idf[related_word] == 0 and idf[word] == 0):
            tf_idf_score = 0
        elif (idf[related_word] == 0):
            tf_idf_score = tf_one * 1/idf[word]
        elif (idf[word] == 0):
            tf_idf_score = tf_two * 1/idf[related_word]
        else:
            tf_idf_score = (tf_one * 1/idf[word]) + (tf_two * 1/idf[related_word])
        if (tf_idf_score > max_score):
            max_index = i
            max_score = score
        if (subj):
            idf_scores.append(tf_idf_score)
            
        words = word_tokenize(sentence)
        if (word not in words or related_word not in words):
            continue
        index_one = words.index(word)
        index_two = words.index(related_word)
        if (index_one != -1 and index_two != -1):
            window_scores.append(abs(index_two - index_one))
            window_sentences.append(i)
    
    if (len(window_scores) != 0):
        min_index = np.argmin(window_scores)
        return JsonResponse({'sentence': sentences[inter[window_sentences[min_index]]]})
    if (len(good_sentence) != 0):
        max_index = np.argmax(idf_scores)
        # return JsonResponse({'sentence': sentences[inter[good_sentence[0]]]})
        return JsonResponse({'sentence': sentences[inter[good_sentence[max_index]]]})
    
    sentence = sentences[inter[max_index]]
    data = {'sentence': sentence}
    return JsonResponse(data)
    