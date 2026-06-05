#!/usr/bin/env python3

import argparse
from ast import arg
import json
import math
import pickle
import string
from InvertedIndex import InvertedIndex  # Fix 1: absolute import
from nltk.stem import PorterStemmer

def removePunc(st):
    return ''.join(ch for ch in st if ch not in string.punctuation)

def getStopWords():
    res = []
    with open("data/stopwords.txt", "r") as file:
        for line in file:
            res.append(removePunc(line.split("\n")[0]))

    return res

STOP_WORDS = getStopWords()


def tokenize(st):
    stop_words = getStopWords()

    tokens = [token.lower() for token in st.split() if token]

    filtered_tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    return filtered_tokens

    


def clean(st):
    stemmer = PorterStemmer()
    no_punc = removePunc(st)
    tokens = tokenize(no_punc)
    clean_tokens = [stemmer.stem(token) for token in tokens]
    return clean_tokens

def tokenize_single_term(st):
    tokens = tokenize(st)

    if len(tokens) != 1:
        raise  ValueError("term must be a single token")
    return tokens[0]



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the index from movies JSON")  # Fix 4: register build

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf",help="Get term frequency")
    tf_parser.add_argument("doc_id",type=int,help="Document Id")
    tf_parser.add_argument("term",type=str,help="Single Term to get its frequency ")

    idf_parser =  subparsers.add_parser("idf",help="Get Inverse Document frequency")
    idf_parser.add_argument("term",type=str,help="Single Term to get its frequency ")

    tf_idf_parser =  subparsers.add_parser("tfidf",help="Get Inverse Document frequency")
    tf_idf_parser.add_argument("doc_id",type=int,help="Document Id")
    tf_idf_parser.add_argument("term",type=str,help="Single Term to get its frequency ")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            index = InvertedIndex()
            with open("cache/index.pkl", "rb") as f:
                index.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                index.docmap = pickle.load(f)

            query = clean(args.query)
            results = {}
            for token in query:
                for doc_id in index.get_documents(token):
                    results[doc_id] = index.docmap[doc_id]
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break

            if not results:
                print("No results found.")
            else:
                for doc_id, movie in results.items():
                    print(f"[{movie['id']}] {movie['title']}")

        case "build":                           # Fix 3: actual build logic
            index = InvertedIndex()
            with open("data/course-rag-movies.json", "r") as f:
                data = json.load(f)
            index.build(data['movies'])
            index.save()
            print("Index built and saved.")
        case "tf":
            token = tokenize_single_term(args.term)
            doc_id = args.doc_id

            index = InvertedIndex()
            with open("cache/term_frequencies.pkl", "rb") as f:
                index.term_frequencies = pickle.load(f)
            print(index.get_tf(doc_id,token))
        case "idf":
            index = InvertedIndex()
            with open("cache/index.pkl", "rb") as f:
                index.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                index.docmap = pickle.load(f)
            token = tokenize_single_term(args.term)

            idf = index.get_idf(token)

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            token = tokenize_single_term(args.term)

            index = InvertedIndex()
            with open("cache/index.pkl", "rb") as f:
                index.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                index.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                index.term_frequencies = pickle.load(f)

            tf = index.get_tf(args.doc_id,token)
            idf = index.get_idf(token)

            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            token = tokenize_single_term(args.term)

            index = InvertedIndex()
            with open("cache/index.pkl", "rb") as f:
                index.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                index.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                index.term_frequencies = pickle.load(f)
            bm25idf = index.get_bm25_idf(token)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()