from summa.keywords import keywords as summa_keywords
from itertools import chain

def match_concepts(my_lib, candidate):
    my_kwds = [keywords(d) for d in my_lib]
    my_kwds = set(chain(*my_kwds))

    candidate_kwds = set(keywords(candidate))

    known_keywords = my_kwds.intersection(candidate_kwds)
    new_keywords = candidate_kwds.difference(known_keywords)
    return known_keywords, new_keywords

def keywords(document):
    assert type(document) is list, "document must be a list"
    try:
        for i in document:
            i.index(" ")
    except ValueError:
        try:
            return summa_keywords(" ".join(document)).split("\n")
        except ZeroDivisionError:
            return []
    else:
        raise ValueError("Expecting a list of strings not a single string.")
