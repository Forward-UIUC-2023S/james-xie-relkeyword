from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from tqdm import tqdm
from parse_csv import parse_csv
import pickle
def extract_sentences(papers):
    noun_occurrence = defaultdict(int)
    pair_cooccur = defaultdict(lambda: defaultdict(int))
    num_nouns = 0
    total_cooccur = defaultdict(int)
    print("Parsing Documents")
    sentences = []

    for i in tqdm(range(len(papers))):
    
    # for i in range(100):
        paper = papers[i]
        # doc = nlp(paper)
        paper_sentences = sent_tokenize(paper)
        sentences.extend(paper_sentences)
        # print (paper_sentences)
    return sentences


documents = parse_csv()

sentences = extract_sentences(documents)
with open('test3/sentences', 'wb') as convert_file:
    pickle.dump(sentences, convert_file)

       