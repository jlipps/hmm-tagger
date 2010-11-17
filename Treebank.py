######### Treebank.py #########

from __future__ import division # use float division
import nltk # import the Natural Language Toolkit for parsing the Penn Treebank
from Helper import msg # for logging
from nltk.corpus.reader import TaggedCorpusReader # use NLTK's corpus reading tools

class Treebank:
    "A class for parsing a tagged corpus for training and testing"
    
    def __init__(self, corpus_path, corpus_files):
        """
        Construct a Treebank object
        
        :param corpus_path: path to corpus files
        :param corpus_files: list of filenames for corpus text
        """

        msg("Importing treebank...")
        
        # get a corpus reader object for our corpus using NLTK
        treebank = TaggedCorpusReader(corpus_path, corpus_files)
        
        # get all sentences from corpus in a tagged format
        self.tagged_sents = treebank.tagged_sents()
        
        # get all sentences from corpus in an untagged format
        self.sents = treebank.sents()
        
        msg("done!\n")
        
    ######### `PUBLIC' FUNCTIONS #########
    
    def training_sents(self, train_pct, start_train_pct):
        """
        Get a list of sentences for training
        
        :param train_pct: what pct of the corpus to retrieve
        :param start_train_pct: where in the corpus to begin retrieval
        """
        
        msg("Getting training sentences...")
        sents = self._sents_by_pct(train_pct, start_train_pct)
        msg("done: %d%% starting at %d%%\n" % (train_pct, start_train_pct))
        
        return sents
        
    def testing_sents(self, test_pct, start_test_pct):
        """
        Get a list of untagged and tagged sentences for testing

        :param test_pct: what pct of the corpus to retrieve
        :param start_test_pct: where in the corpus to begin retrieval
        """
        
        # when we retrieve testing sentences, we want to get tagged and untagged
        # versions of them so we can evaluate accuracy
        msg("Getting testing sentences...")
        untagged_sents = self._sents_by_pct(test_pct, start_test_pct, tagged=False)
        tagged_sents = self._sents_by_pct(test_pct, start_test_pct, tagged=True)
        msg("done: %d%% starting at %d%%\n" % (test_pct, start_test_pct))
        
        return (untagged_sents, tagged_sents)
        
    def pos_tags(self):
        """
        Create a list of all POS tags found in the corpus
        """
        
        msg("Getting POS tag list...")
        tags = []
        
        # loop through sentences
        for sent in self.tagged_sents:
            
            # loop through tagged words
            for (word, pos) in sent:
                
                # add tag if it's not already in list
                if pos not in tags:
                    tags.append(pos)

        msg("done\n")
        
        return tags
        
    
    ######### `PRIVATE' FUNCTIONS #########
    
    def _sents_by_pct(self, pct, start_pct, tagged=True):
        """
        Retrieve a percentage of sentences from the corpus
        
        :param pct: what pct of the corpus to retrieve
        :param start_pct: what point in the corpus to begin retrieval
        :param tagged: whether to return tagged words (default: True)
        """
        
        # choose the corpus sentence list based on tagged
        if tagged:
            tb_sents = self.tagged_sents
        else:
            tb_sents = self.sents
        
        total_sents = len(tb_sents)
        last_index = total_sents - 1
        end_pct = pct + start_pct
        
        # if our end_pct is greater than 100, go around the 0% corner
        if end_pct > 100:
            end_pct -= 100
        
        # get the index of the first sentence we want
        first_sent_index = int(total_sents * start_pct / 100)
        
        # get the index of the last sentence we want:
        # subtract 1 to make sure testing/training sentence lists do not overlap
        last_sent_index = int(total_sents * end_pct / 100) - 1
        
        # if the last index is one less than the actual last index, push it up to the
        # end (to account for the behavior of last_sent_index)
        if last_sent_index == last_index - 1:
             last_sent_index = last_index
             
        # retrieve the sentences based on the indices we calculated
        return self._sents_by_range(tb_sents, first_sent_index, last_sent_index)
        
    def _sents_by_range(self, tb_sents, first_sent_index, last_sent_index):
        """
        Retreive a subset of a list of sentences based on index range
        
        :param tb_sents: the list of sentences to retrieve from
        :param first_sent_index: the index of the first sentence in the range we want
        :param last_sent_index: the index of the last sentence in the range we want
        """
        
        # if our last index is smaller than our first, we need to take 2 slices
        if last_sent_index < first_sent_index:
            # get sentences from first index to end, then from beginning to second
            # index
            sents = tb_sents[first_sent_index:len(tb_sents)] + \
                tb_sents[0:last_sent_index+1]
            
        # otherwise, we perform a simple range slice
        else:
            sents = tb_sents[first_sent_index:last_sent_index+1]
            
        return sents        