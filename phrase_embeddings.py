from sentence_transformers import SentenceTransformer
import json
from collections import defaultdict
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
# cos_sim = nn.CosineSimilarity(dim=0)

# model_path = './phrase-bert-model/pooled_context_para_triples_p=0.8/0_BERT/'
model = SentenceTransformer('whaleloops/phrase-bert')
print("Retrieving Words")
with open("yake_output.json") as json_file:
    data = json.load(json_file)
    candidate_words = []
    for i in data:
        if (i[1] >= 50):
            candidate_words.append(i[0].lower())
print ("Encoding")       
phrase_embs = model.encode( candidate_words )
print ("calculaating similarity")

semantic_similar = defaultdict(lambda: defaultdict(int))

index = 0
for phrase, embedding in tqdm(zip(candidate_words, phrase_embs)):
    
    for other_phrase, other_embedding in zip(candidate_words, phrase_embs):
        if (phrase is not other_phrase):
            # semantic_similar[phrase][other_phrase] = cos_sim( torch.tensor(embedding), torch.tensor(other_embedding))
            semantic_similar[phrase][other_phrase] = str(dot(embedding, other_embedding)/(norm(embedding) * norm(other_embedding)))

    
copy = defaultdict(int)
print ("READY TO PRINT")
with open('test3/presortedsemantic_similar2.json', 'w') as convert_file:
    convert_file.write(json.dumps((semantic_similar)))
    
for key in semantic_similar:
    copy[key] = sorted(semantic_similar[key].items(), key = lambda x: x[1])[-10:]
with open('test3/semantic_similar2.json', 'w') as convert_file:
    convert_file.write(json.dumps((copy)))