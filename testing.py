from relkeyword import CorpusAnalyzer, KeywordSuggester

from parse_csv import parse_csv
documents = parse_csv()
documents = documents[:1000]
c = CorpusAnalyzer.CorpusAnalyzer(documents, "grants2")
c.generate_candidate_keyphrases(threshold = 5)
c.generate_related_keywords()
c.generate_semantic_similar()


kw_suggester = KeywordSuggester.KeywordSuggester("relkeyword_dir/grants2/")
print(kw_suggester.retrieve_related_words("children"))
print(kw_suggester.retrieve_sentence("children", "youth"))