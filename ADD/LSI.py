from rake_nltk import Rake
from nltk import pos_tag
from itertools import compress
from itertools import chain
from tools.DocumentSimilarity import DocumentSimilarity
import numpy as np
import scipy

def filter_by_pos(keyphrases, min_len=0):
    """
    filters keyphrases by a pos pattern.
    """
    def has_only_desired_pos(keyphrases):
        """
        takes a list of keyphrases (ie strings).
        returns a boolean list of size `len(keywords)`. if `True` 
        the corresponding keyphrase consists only of nouns and 
        adjectives. Otherwise, `False` is returned.
        """
        def is_desired_pos_tag(tag):
            return True if tag[:2] in ("NN", "JJ") else False

        POSed_tokens = [pos_tag(keyphrase.split()) for keyphrase 
            in keyphrases]
        has_desired_pos = [[is_desired_pos_tag(tag) 
            for keyword, tag in keyphrase] 
                for keyphrase in POSed_tokens]
        has_only_desired_pos = [all(b) for b in has_desired_pos]

        return has_only_desired_pos

    # filter keyphrases by pos tags
    desired = has_only_desired_pos([k for _, k in keyphrases])
    filtered_keyphrases = list(compress(keyphrases, desired))

    # exclude one token keyphrases
    filtered_keyphrases = [k for s, k in filtered_keyphrases 
                             if len(k.split()) > min_len]
    return filtered_keyphrases

def keyphrases(text, mu=2, sig=1.5):
    """
    determines and ranks keyphrases from `text`. the keyphrases are
    weighted such that short keyphrases (2-3 words) are preferred. 
    moveover, keyphrases not adhering the rules defined in
    `filter_by_pos` are abandoned. 
    """
    assert type(text) is list, "the text is not a list"
    r = Rake(punctuations=". , ? ! - : ; \" \' ( )".split(), language='english')
    try:
        text[0].index(" ")
    except ValueError:
        pass
    else:
        raise ValueError("expecting a list a strings not a single string")

    text = " ".join(text)
    r.extract_keywords_from_text(text)
    
    # the scores are weighted by their length (# tokens)
    # using a normal distribution
    n = scipy.stats.norm(mu, sig)
    scores = r.get_ranked_phrases_with_scores()
    scores = [(s * n.pdf(len(f.split())), f) 
        for s, f in scores]
    
    scores = sorted(scores, key=lambda x: -x[0])
    return filter_by_pos(scores)

def match_concepts(my_lib, candidate, sim_threshold=.8):
    """
    Does a fuzzy compare between the concepts in `my_lib` and
    `candidate`. returns two sets:
        (1) the set of common concepts
        (2) the set of new concepts
    `sim_threshold` is the similarity boundry that defines 
    whether two concepts are equal or not.
     
    optimization oportunities:
    * currently only the most similar keyphrase determines the 
    decisionmaking. more keyphrases could be used.
    """
    
    def most_similar(corpus, query):
        """
        returns the keywords in `corpus` ordered by similarity to 
        `query`.
        """
        comparator = DocumentSimilarity()
        sims = comparator(corpus=corpus, query=query)
        return [(sims[idx], idx) for idx in np.argsort(sims)[::-1]]
    
    # determine the keyphrases of "my library"
    my_keyphrases = {p for p in chain(
        *[keyphrases(t) for t in my_lib]
    )}
    
    # determine the candidates keyphrases
    candidate_keyphrases = keyphrases(candidate)

    # decide whether keyphrases are similar to known keyphrases
    known_concepts = set()
    new_concepts = set()
    my_keyphrases = [p.split() for p in my_keyphrases]
    for candidate_keyphrase in candidate_keyphrases:
        similarities = most_similar(my_keyphrases, 
                                    candidate_keyphrase.split())
        if similarities[0][0] >= sim_threshold:
            known_concepts.add(candidate_keyphrase)
        else:
            new_concepts.add(candidate_keyphrase)
    return known_concepts, new_concepts

keywords = keyphrases # for API reasons
