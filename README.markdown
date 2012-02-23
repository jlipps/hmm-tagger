hmm-tagger
===

This is a [Part of Speech](http://en.wikipedia.org/wiki/Part-of-speech_tagging) tagger written in Python, utilizing the [Viterbi algorithm](http://en.wikipedia.org/wiki/Viterbi_algorithm) (an instantiation of [Hidden Markov Models](http://en.wikipedia.org/wiki/Hidden_Markov_model)). It uses the Natural Language Toolkit and trains on Penn Treebank-tagged text files. It will use ten-fold cross validation to generate accuracy statistics, comparing its tagged sentences with the gold standard.

Usage
---
    python hmm-tagger.py [--clean]

Pass in the --clean option to clean a Treebank file before running the tagger. This can be time consuming, so you can leave it off during future runs.
