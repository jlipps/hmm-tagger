######### clean.py #########

from TreebankCleaner import TreebankCleaner # import cleaning class

# initialize treebank cleaner with the appropriate path and file(s)
t = TreebankCleaner('/Users/jlipps/Code/hmm-tagger/', ['treebank3_sect2.txt'])

t.clean()