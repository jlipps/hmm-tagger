######### hmm-tagger.py #########

from TreebankCleaner import TreebankCleaner # import cleaning class
from Tagger import Tagger # import the tagging controller

# initialize treebank cleaner with the appropriate path and file(s)
t = TreebankCleaner('/Users/jlipps/Code/hmm-tagger/', ['treebank3_sect2.txt'])

# do cleaning
t.clean()

# initialize a tagging object with the cleaned corpus file(s)
t = Tagger('/Users/jlipps/Code/hmm-tagger/', ['treebank3_sect2.txt_cleaned'])

# perform ten-fold cross-validation
t.run_test_cycles()