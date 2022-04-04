#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYU NLP Homework 6: Feature selection for Maxent semantic role labeling.
    by Ziyang Zeng (zz2960)
    Spring 2022
"""


import argparse
from typing import TypedDict, Union, List
import nltk
from fastprogress.fastprogress import progress_bar

stemmer = nltk.stem.SnowballStemmer('english')


class Word(TypedDict):
    word: str
    pos: str
    biotag: str
    capitalized: bool
    label: Union[str, None]


def parse_input(input_file: str) -> List[Union[List[Word], None]]:
    """
    Parses the input file and returns a list of lists of words.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()
    sentences: List[Union[List[Word], None]] = []
    last_sentence: List[Word] = []
    print("Parsing input file lines...")
    for line in progress_bar(lines):
        line = line.strip()
        word_info = line.split("\t")
        if len(word_info) >= 5:
            word_str = word_info[0].strip()
            capitalized = word_str[0].isupper()
            pos = word_info[1].strip()
            biotag = word_info[2].strip()
            if len(word_info) >= 6:
                label = word_info[5].strip()
            else:
                label = None
            word = Word(word=word_str, pos=pos, biotag=biotag,
                        capitalized=capitalized, label=label)
            last_sentence.append(word)
        else:
            if len(last_sentence) > 0:
                sentences.append(last_sentence)
            last_sentence = []
            sentences.append(None)
    if len(last_sentence) > 0:
        sentences.append(last_sentence)
    return sentences


class WordFeatures(TypedDict):
    word: str
    stem: str
    pos: str
    biotag: str
    position: float
    previous_tag: Union[str, None]
    previous_pos: Union[str, None]
    previous_biotag: Union[str, None]
    previous_word: Union[str, None]
    previous_stem: Union[str, None]
    previous_2_pos: Union[str, None]
    previous_2_biotag: Union[str, None]
    previous_2_word: Union[str, None]
    previous_2_stem: Union[str, None]
    previous_3_pos: Union[str, None]
    previous_3_biotag: Union[str, None]
    previous_3_word: Union[str, None]
    previous_3_stem: Union[str, None]
    next_pos: Union[str, None]
    next_biotag: Union[str, None]
    next_word: Union[str, None]
    next_stem: Union[str, None]
    next_2_pos: Union[str, None]
    next_2_biotag: Union[str, None]
    next_2_word: Union[str, None]
    next_2_stem: Union[str, None]
    next_3_pos: Union[str, None]
    next_3_biotag: Union[str, None]
    next_3_word: Union[str, None]
    next_3_stem: Union[str, None]
    capitalized: bool
    label: Union[str, None]


def get_word_features(sentence: List[Word]) -> List[WordFeatures]:
    """
    Returns a list of word features for a sentence.
    """
    word_features = []
    sentence_len = len(sentence)
    for i in range(sentence_len):
        word = sentence[i]
        position = i / sentence_len
        word_str = word["word"]
        word_stem = stemmer.stem(word_str)
        word_pos = word["pos"]
        word_biotag = word["biotag"]
        word_capitalized = word["capitalized"]
        word_label = word["label"]
        if i >= 1:
            previous_tag = "@@"
            previous_word = sentence[i-1]
            previous_word_str = previous_word["word"]
            previous_word_pos = previous_word["pos"]
            previous_word_biotag = previous_word["biotag"]
            previous_word_stem = stemmer.stem(previous_word_str)
        else:
            previous_tag = None
            previous_word_str = None
            previous_word_pos = None
            previous_word_biotag = None
            previous_word_stem = None
        if i >= 2:
            previous_2_word = sentence[i-2]
            previous_2_word_str = previous_2_word["word"]
            previous_2_word_pos = previous_2_word["pos"]
            previous_2_word_biotag = previous_2_word["biotag"]
            previous_2_word_stem = stemmer.stem(previous_2_word_str)
        else:
            previous_2_word_str = None
            previous_2_word_pos = None
            previous_2_word_biotag = None
            previous_2_word_stem = None
        if i >= 3:
            previous_3_word = sentence[i-3]
            previous_3_word_str = previous_3_word["word"]
            previous_3_word_pos = previous_3_word["pos"]
            previous_3_word_biotag = previous_3_word["biotag"]
            previous_3_word_stem = stemmer.stem(previous_3_word_str)
        else:
            previous_3_word_str = None
            previous_3_word_pos = None
            previous_3_word_biotag = None
            previous_3_word_stem = None
        if i <= sentence_len - 2:
            next_word = sentence[i+1]
            next_word_str = next_word["word"]
            next_word_pos = next_word["pos"]
            next_word_biotag = next_word["biotag"]
            next_word_stem = stemmer.stem(next_word_str)
        else:
            next_word_str = None
            next_word_pos = None
            next_word_biotag = None
            next_word_stem = None
        if i <= sentence_len - 3:
            next_2_word = sentence[i+2]
            next_2_word_str = next_2_word["word"]
            next_2_word_pos = next_2_word["pos"]
            next_2_word_biotag = next_2_word["biotag"]
            next_2_word_stem = stemmer.stem(next_2_word_str)
        else:
            next_2_word_str = None
            next_2_word_pos = None
            next_2_word_biotag = None
            next_2_word_stem = None
        if i <= sentence_len - 4:
            next_3_word = sentence[i+3]
            next_3_word_str = next_3_word["word"]
            next_3_word_pos = next_3_word["pos"]
            next_3_word_biotag = next_3_word["biotag"]
            next_3_word_stem = stemmer.stem(next_3_word_str)
        else:
            next_3_word_str = None
            next_3_word_pos = None
            next_3_word_biotag = None
            next_3_word_stem = None
        features = WordFeatures(word=word_str, stem=word_stem, pos=word_pos, biotag=word_biotag,
                                position=position,
                                previous_tag=previous_tag,
                                previous_pos=previous_word_pos,
                                previous_biotag=previous_word_biotag,
                                previous_word=previous_word_str,
                                previous_stem=previous_word_stem,
                                previous_2_pos=previous_2_word_pos,
                                previous_2_biotag=previous_2_word_biotag,
                                previous_2_word=previous_2_word_str,
                                previous_2_stem=previous_2_word_stem,
                                previous_3_pos=previous_3_word_pos,
                                previous_3_biotag=previous_3_word_biotag,
                                previous_3_word=previous_3_word_str,
                                previous_3_stem=previous_3_word_stem,
                                next_pos=next_word_pos,
                                next_biotag=next_word_biotag,
                                next_word=next_word_str,
                                next_stem=next_word_stem,
                                next_2_pos=next_2_word_pos,
                                next_2_biotag=next_2_word_biotag,
                                next_2_word=next_2_word_str,
                                next_2_stem=next_2_word_stem,
                                next_3_pos=next_3_word_pos,
                                next_3_biotag=next_3_word_biotag,
                                next_3_word=next_3_word_str,
                                next_3_stem=next_3_word_stem,
                                capitalized=word_capitalized,
                                label=word_label)
        word_features.append(features)
    return word_features


def main():
    parser = argparse.ArgumentParser(
        description="A feature selector for Maxent semantic role labeling")
    parser.add_argument("-s", "--strip", help="strip label for development or testing",
                        action="store_true", default=False)
    parser.add_argument("inputfile", help="input corpus file")
    parser.add_argument("outfile", help="feature selection output file")

    args = parser.parse_args()

    sentences = parse_input(args.inputfile)

    print("\nSelecting features...")
    with open(args.outfile, "w") as f:
        for sentence in progress_bar(sentences):
            if sentence is None:
                f.write("\n")
            else:
                word_features = get_word_features(sentence)
                for word_feature in word_features:
                    word_feature_str_list = []
                    for key, value in word_feature.items():
                        if value is None or key == "word" or key == "label":
                            continue
                        word_feature_str_list.append(f"{key.upper()}={value}")
                    word_feature_str_list.insert(0, word_feature["word"])
                    if not args.strip:
                        if word_feature["label"] is not None:
                            word_feature_str_list.append(word_feature['label'])
                        else:
                            word_feature_str_list.append("")
                    f.write("\t".join(word_feature_str_list))
                    f.write("\n")
        print(f"{args.inputfile} -> {args.outfile}.")


if __name__ == '__main__':
    main()
