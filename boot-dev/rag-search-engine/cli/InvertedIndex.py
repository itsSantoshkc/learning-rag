from collections import Counter, defaultdict
import json
import os
import pickle
import string
from nltk.stem import PorterStemmer


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)           # token -> set(doc_id)
        self.docmap = {}                        # doc_id -> document object
        self.term_frequencies = defaultdict(Counter)
        self.stop_words = self.getStopWords()
        self.stemmer = PorterStemmer()          # Fix 1: instantiate once

    def __removePunc(self, st):
        return ''.join(ch for ch in st if ch not in string.punctuation)

    def getStopWords(self):
        res = []
        with open("data/stopwords.txt", "r") as file:
            for line in file:
                res.append(self.__removePunc(line.split("\n")[0]))
        return res

    def __tokenize(self, text):
        new_text = self.__removePunc(text)
        tokens = [token.lower() for token in new_text.split() if token]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]  # Fix 1: use self.stemmer
        return tokens

    def __add_document(self, doc_id, text):
        tokens = self.__tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)       # Fix 2: removed redundant check
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def get_tf(self, doc_id, term):
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term, 0)

    def build(self, movies):
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            combined_text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, combined_text)

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)


if __name__ == "__main__":                      # Fix 3: consolidated under guard
    index = InvertedIndex()
    with open("data/course-rag-movies.json", "r") as f:
        data = json.load(f)
    index.build(data['movies'])
    index.save()