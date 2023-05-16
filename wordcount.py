# import pandas as pd
# df = pd.read_csv('unigram_freq.csv', header=None, index_col=0, squeeze = True)
# d = df.to_dict()
# print(d)    
from collections import defaultdict
import json
with open('test3/semantic_related.json') as f:
    data = f.read()
    
js = json.loads(data)

# for value in js.values(): # use `itervalues` In Python 2.x
    
#     value.sort(key=lambda x: int(x['order']))
copy1 = defaultdict(lambda: defaultdict(int))
for key in js:
    print (js[key])
    for pair in js[key]:
        if (key is pair or pair in key):
            continue
        copy1[key][pair] = js[key][pair]
copy = defaultdict(int) 

for key in js:
    copy[key] = sorted(copy1[key].items(), key = lambda x: x[1])[-20:]
    copy[key] = copy[key][::-1]


# domain_relevance = sorted(js.items(), key=lambda x: x[1])

with open('test3/sorted_semantic_related2.json', 'w') as convert_file:
    convert_file.write(json.dumps(copy))
        
# with open('co_occurrence.txt') as f:
#     data = f.read()
    
# js = json.loads(data)
# co_occurrence = sorted(js.items(), key=lambda x: x[1])

# with open('co_occurrence.txt', 'w') as convert_file:
#     convert_file.write(json.dumps(co_occurrence))
        
        
# with open('embedding_similarity.txt') as f:
#     data = f.read()
    
# js = json.loads(data)
# embedding_similarity = sorted(js.items(), key=lambda x: x[1])

# with open('embedding_similarity.txt', 'w') as convert_file:
#     convert_file.write(json.dumps(embedding_similarity))
        

# with open('vertex_weights.txt') as f:
#     data = f.read()
    
# js = json.loads(data)
# co_occurrence = sorted(js.items(), key=lambda x: x[1])

# with open('vertex_weights.txt', 'w') as convert_file:
#     convert_file.write(json.dumps(co_occurrence))
        