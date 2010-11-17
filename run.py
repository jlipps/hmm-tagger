######### run.py #########

from Tagger import Tagger # import the tagging controller

# initialize a tagging object with the cleaned corpus file(s)
t = Tagger('/Users/jlipps/Code/hmm-tagger/', ['treebank3_sect2.txt_cleaned'])

# perform ten-fold cross-validation
t.run_test_cycles()