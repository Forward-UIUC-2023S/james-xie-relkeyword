from collections import defaultdict
import json
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")

with open('test3/semantic_similar2.json') as f:
    semantic_similar = f.read()
    
semantic_similar = json.loads(semantic_similar)

with open('test3/sorted_semantic_related.json') as f:
    semantic_related = f.read()
    
semantic_related = json.loads(semantic_related)


with open('test3/word_indexing.json') as f:
    word_indexing = f.read()
    
word_indexing = json.loads(word_indexing)

with open('test3/sentences', 'rb') as f:
    sentences = pickle.load(f)
    
with open('test2/tf.json') as f:
    tf = f.read()
tf = json.loads(tf)

with open('test2/idf.json') as f:
    idf = f.read()
idf = json.loads(idf)

with open('test2/sentence_to_doc.json') as f:
    sentence_to_doc = f.read()
sentence_to_doc = json.loads(sentence_to_doc)

# word_indexing = json.loads(word_indexing)
# for value in js.values(): # use `itervalues` In Python 2.x
    
#     value.sort(key=lambda x: int(x['order']))
while (True):
    user_input = input("Enter Keyword: ")
    if (user_input not in semantic_similar):
        print("Word does not appear in corpus as a noun or adjective")
        continue
    related_words = list(reversed(semantic_related[user_input]))
    similar_words = list(reversed(semantic_similar[user_input]))
    print("RELATED WORDS: ")
    print(related_words)
    
    print ("SIMILAR WORDS: ")
    print (similar_words)
    
    
    first_word = input("Is there a recommended keyword you would like to learn more about? (Y/N) ")
    if (first_word == "N"):
        continue
    while (True):
        suggested_word = input("Which recommended keyword? ")
        # first_word = related_words[0][0]
        related_sentences = word_indexing[suggested_word]
        curr_sentences = word_indexing[user_input]
        if (len(related_sentences) > len(curr_sentences)):
            inter = set(related_sentences).intersection(curr_sentences)
        else: 
            inter = set(curr_sentences).intersection(related_sentences) 
        inter = list(inter)
        scores = []
        max_score = 0
        max_index = 0
        
#         for i in range(len(inter)):
#             index = inter[i]
#             sentence = sentences[index]
#             # print(sentence_to_doc[str(index)])
#             score = 0
#             term_frequencies = tf[str(sentence_to_doc[str(index)])]
#             sum = 0
#             # print (sentence)
#             doc = nlp(sentence)
#             # split_sentence = sentence.split(" ")
#             for word in doc:
                
#                 # print(word)
#                 word = word.text.lower()
#                 if (word not in term_frequencies):
#                     continue
#                 # term_frequencies[word]
#                 # idf[word]
#                 tf_idf_score = term_frequencies[word] * 1/idf[word]
#                 sum += tf_idf_score
#             if (score > max_score):
#                 max_index = i
#                 max_score = score
#         print()
#         print(sentences[inter[max_index]])
#         print()
#         # i = 0
        
#         continue_recommended = input("Would you like to see another recommended keyword? (Y/N): ")
#         if (continue_recommended == "N"):
#             break
#         # print ()
#         # print (sentences[inter[i]])
#         # print ()
#         # while (True):
            
#         #     continuing = input("Would you like to see another sentence? (Y/N) ")
#         #     if (continuing == "N"):
#         #         break
#         #     i += 1
#         #     print (sentences[inter[i]])
        
    
#         # print (inter)
#     # output = set()
#     # for index in inter:
#     #     print (len(sentences[index]))
#     #     if (sentences[index] != ""):
#     #         output.add(sentences[index])
#     # for sentence in output:
#     #     print(sentence)
# # copy = defaultdict(int)

# # for key in js:
# #     copy[key] = sorted(js[key].items(), key = lambda x: x[1])[-10:]
    
# # # domain_relevance = sorted(js.items(), key=lambda x: x[1])

# # with open('test2/sorted_semantic_similar.txt', 'w') as convert_file:
# #     convert_file.write(json.dumps(copy))