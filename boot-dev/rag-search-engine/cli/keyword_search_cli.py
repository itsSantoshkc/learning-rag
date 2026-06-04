#!/usr/bin/env python3

import argparse
import json
import string
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

    tokens = [token for token in st.split() if token]

    filtered_tokens = [
        token for token in tokens
        if token.lower() not in stop_words
    ]

    return filtered_tokens


def clean(st):
    stemmer = PorterStemmer()
    no_punc = removePunc(st)
    tokens = tokenize(no_punc)
    clean_tokens = [stemmer.stem(token) for token in tokens]
    return clean_tokens



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            with open("data/course-rag-movies.json", "r") as f:
                d = json.load(f)
            res = []
            for movie in d["movies"]:
                if(len(res) >= 5):
                    break
                clean_tokens = clean(movie["title"])
                clean_query_tokens = clean(args.query)


                for token in clean_tokens:
                    for qtoken in clean_query_tokens:
                        if qtoken in token.lower() :
                            if(movie["title"] not in res):
                                res.append(movie["title"])
                            

                
            print(res)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
