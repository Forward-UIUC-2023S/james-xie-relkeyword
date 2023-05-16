import sent2vec
import json
from tqdm import tqdm
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm


import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader as api


with open("yake_output.json") as json_file:
    data = json.load(json_file)
    candidate_words = []
    for i in data:
        if (i[1] > 10):
            candidate_words.append(i[0].lower())
# model = sent2vec.Sent2vecModel()
# model.load_model('model2.bin')
# embs = model.embed_sentences(candidate_words)
# semantic_similar = defaultdict(lambda: defaultdict(int))
# for phrase, embedding in tqdm(zip(candidate_words, embs)):
    
#     for other_phrase, other_embedding in zip(candidate_words, embs):
#         if (phrase is not other_phrase):
#             # semantic_similar[phrase][other_phrase] = cos_sim( torch.tensor(embedding), torch.tensor(other_embedding))
#             semantic_similar[phrase][other_phrase] = str(dot(embedding, other_embedding)/(norm(embedding) * norm(other_embedding)))
# print (embs)
# copy = defaultdict(int)
# for key in semantic_similar:
#     copy[key] = sorted(semantic_similar[key].items(), key = lambda x: x[1])[-20:]
# with open('test3/semantic_similar_sent2vec.json', 'w') as convert_file:
#     convert_file.write(json.dumps((copy)))
import numpy as np
model = gensim.models.Word2Vec.load("word2vec2.model")
word_vectors = model.wv
print (word_vectors.key_to_index)
embeddings = []
for word in candidate_words:
    # print (word)
    words = word.split()
    curr_embeddings = []
    for w in words:
        # print (w)
        if word not in model.wv.key_to_index:
            continue
        print ("appending")
        curr_embeddings.append(model.wv[w])
    
    embeddings.append(np.mean(curr_embeddings))
semantic_similar = defaultdict(lambda: defaultdict(int))

for phrase, embedding in tqdm(zip(candidate_words, embeddings)):
    
    for other_phrase, other_embedding in zip(candidate_words, embeddings):
        if (phrase is not other_phrase):
            
            # semantic_similar[phrase][other_phrase] = cos_sim( torch.tensor(embedding), torch.tensor(other_embedding))
            if (norm(embedding) * norm(embedding) != 0):
                semantic_similar[phrase][other_phrase] = str(dot(embedding, other_embedding)/(norm(embedding) * norm(other_embedding)))
with open("word2vec_embed.json", 'w') as f:
    f.write(json.dumps(semantic_similar))
copy = defaultdict(int)
for key in semantic_similar:
    copy[key] = sorted(semantic_similar[key].items(), key = lambda x: x[1])[-20:]
with open('test3/word2vec_similar.json', 'w') as convert_file:
    convert_file.write(json.dumps((copy)))