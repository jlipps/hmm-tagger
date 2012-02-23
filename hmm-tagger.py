######### hmm-tagger.py #########

from TreebankCleaner import TreebankCleaner # import cleaning class
from Tagger import Tagger # import the tagging controller
import os # for path info
import sys # for command line options

if '--clean' in sys.argv:
  # initialize treebank cleaner with the current path and pre-downloaded file(s)
  t = TreebankCleaner(os.getcwd()+'/', ['treebank3_sect2.txt'])
  # do cleaning
  t.clean()

# initialize a tagging object with the cleaned corpus file(s)
t = Tagger(os.getcwd()+'/', ['treebank3_sect2.txt_cleaned'])

# perform ten-fold cross-validation
t.run_test_cycles()
