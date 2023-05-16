import re
import string
import nltk
import json
from parse_csv import parse_csv
from tqdm import tqdm 
from nltk.tokenize import word_tokenize, sent_tokenize

with open("yake_output.json") as json_file:
    data = json.load(json_file)
    candidate_words = []
    for i in data:
        if (i[1] > 10):
            candidate_words.append(i[0].lower())
            
# with open('sentences.txt', 'r') as f:
#     sentences = f.read()
papers = parse_csv()
sentences = []
for i in tqdm(range(len(papers))):
    
    # for i in range(100):
        paper = papers[i]
        paper = paper.lower()
        # doc = nlp(paper)
        paper_sentences = sent_tokenize(paper)
        for sentence in paper_sentences:
            words = word_tokenize(sentence)
            sentences.append(words)
        # print (paper_sentences)
    
# new_sentences = []
# print ("parsing sentences")
# for sentence in sentences:
#     sentence = re.sub(r"http\S+", "", sentence)
#     sentence = re.sub("[^A-Za-z]+", " ", sentence)
#     new_sentences.append(sentence)
    
from gensim.models import Word2Vec
print ("training word2vec")
model = Word2Vec(sentences=sentences)
model.save("word2vec2.model")

# with open('new_sentences.txt', 'w') as f:
#     f.write(str(new_sentences))