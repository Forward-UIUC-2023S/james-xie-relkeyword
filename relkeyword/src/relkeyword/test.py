from CorpusAnalyzer import CorpusAnalyzer
from KeywordSuggester import KeywordSuggester
from nltk.tokenize import word_tokenize, sent_tokenize
import sent2vec
import csv
# def parse_csv():
#     f = open("/Users/jamesxie/Desktop/FWD Data Lab/RelatedKeyword/relkeyword/src/relkeyword/Grants-20230219.csv", "r+")
#     t = f.readlines()
#     text = ""
#     for i in t:
#         text += i.replace('\0', '')
#     f.seek(0)
#     f.write(text)
#     f.seek(0)
#     csv_reader = csv.reader(f)

#     # use the csv file
#     papers = []
#     for row in csv_reader:
#         if (row == []):
#             continue
#         papers.append(str(row[1] + ". " + row[3]))

#     f.close()
#     return papers

# papers = parse_csv()
# sentences = []
# from tqdm import tqdm

# # for i in tqdm(range(1000)):
# #     paper = papers[i]
# #     paper = paper.lower()
# #     paper_sentences = sent_tokenize(paper)
# #     sentences.extend(paper_sentences)
    
# # with open(r'./sentences.txt', 'w') as fp:
# #     for sentence in sentences:
# #         fp.write("%s\n" %sentence)
# papers = papers[:1000]
# corpus_analyzer = CorpusAnalyzer(papers, "testing")
# corpus_analyzer.generate_candidate_keyphrases(threshold=5)
# corpus_analyzer.generate_related_keywords()
# corpus_analyzer.generate_semantic_similar()
kw = KeywordSuggester("relkeyword_dir/testing/")
print(kw.retrieve_related_words("children"))
print (kw.retrieve_sentence("children", "youth"))