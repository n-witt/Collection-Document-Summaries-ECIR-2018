from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain

class TFIDF:
    def __init__(self, corpus):    
        self.corpus = [" ".join(d) for d in corpus]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(self.corpus)
        
    def _tfidfed_keywords(self, document):
        """
        finds words and their corresponding tfidf scores in `document`.
        """
        keywords = dict()
        transformed_doc = self.vectorizer.transform([" ".join(document)]).toarray()[0]
        for w in document:
            try:
                word_id = self.vectorizer.vocabulary_[w]
                keywords[w] = transformed_doc[word_id]
            except KeyError:
                pass
        return keywords


    def keywords(self, document, fraction=0.2):
        """
            extracts keywords based on their tfidf score. ie the words 
            with the highest tfidf score are selected. 
            `fraction` determines the percentage of words that will be 
            selected from the sorted list (eg .2 means the top 20% are
            selected).
            """
        assert type(document) is list, "document is not a list of strings"
        try:
            document[0].index(" ")
        except ValueError:
            keywords = self._tfidfed_keywords(document)
            last_idx = int(len(keywords) * fraction)
            clipped_keywords = sorted(keywords.items(), 
                                      key=lambda x: x[1], reverse=True)[0:last_idx]
            return [x[0] for x in clipped_keywords]
        else:
            raise ValueError("expecting a list of strings not a single string")


    def match_concepts(self, my_lib, candidate):
        my_keywords = set(chain(*[self.keywords(d) for d in my_lib]))
        candidate_kws = set(self.keywords(candidate))
        
        known_kws = my_keywords.intersection(candidate_kws)
        new_kws = candidate_kws.difference(my_keywords)
        
        return known_kws, new_kws

