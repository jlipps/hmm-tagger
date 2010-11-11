from Tagger import Tagger

t = Tagger('/Users/jlipps/Code/hmm-tagger/', ['treebank3_sect2.txt_cleaned'])
t.run_test_cycles()