NYU NLP Homework 6: Build a semantic role labeling system
    by Ziyang Zeng (zz2960)
    Spring 2022

I reused the Maxent model from the previous homework 5.

Features selected:
    - word stem
    - the position (percentage) of the word in the sentence
    - POS tag
    - BIO tag
    - whether capitalized
    - previous word
    - previous POS tagger
    - previous BIO tag
    - previous previous word
    - previous previous POS tag
    - previous previous BIO tag
    - next word
    - next POS tag
    - next BIO tag
    - next next word
    - next next POS tag
    - next next BIO tag

For scoring, I used the following metrics:
    - precision
    - recall
    - F1

On dev set, I achieved:
    - precision:   0.7429906542056075
    - recall:      0.7004405286343612
    - F1:          0.7210884353741497
