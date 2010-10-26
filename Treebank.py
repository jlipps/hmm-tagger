from __future__ import division # use float division
import nltk # we need to import the Natural Language Toolkit for parsing the Penn Treebank

class Treebank:
    "A class for parsing the Penn Treebank for training and testing"
    
    def __init__(self, train_pct=90, test_pct=10, start_train_pct=0):
        from nltk.corpus import treebank
        self.tb = treebank
        self.tagged_sents = self.tb.tagged_sents()
        self.sents = self.tb.sents()
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.start_train_pct = start_train_pct
        
    def training_sents(self, train_pct=False, start_train_pct=False):
        train_pct = train_pct or self.train_pct
        if start_train_pct is False:
            start_train_pct = self.start_train_pct
        # print "Training %%: %d%%" % train_pct
        # print "Start training at: %d%%" % start_train_pct
        return self._sents_by_pct(train_pct, start_train_pct)
        
    def testing_sents(self, test_pct=False, start_test_pct=False):
        test_pct = test_pct or self.test_pct
        if start_test_pct is False:
            start_test_pct = self.start_train_pct + self.train_pct
        # print "Testing %%: %d%%" % test_pct
        # print "Start testing at: %d%%" % start_test_pct
        untagged_sents = self._sents_by_pct(test_pct, start_test_pct, tagged=False)
        tagged_sents = self._sents_by_pct(test_pct, start_test_pct, tagged=True)
        return (untagged_sents, tagged_sents)
        
    def pos_tags(self):
        tags = []
        for sent in self.tagged_sents:
            for (word, pos) in sent:
                if pos not in tags:
                    tags.append(pos)
        return tags
        
    
    # `PRIVATE' FUNCTIONS
    
    def _sents_by_pct(self, pct, start_pct, tagged=True):
        if tagged:
            tb_sents = self.tagged_sents
        else:
            tb_sents = self.sents
        total_sents = len(tb_sents)
        last_index = total_sents - 1
        end_pct = pct + start_pct
        if end_pct > 100:
            end_pct -= 100
        first_sent_index = int(total_sents * start_pct / 100)
        last_sent_index = int(total_sents * end_pct / 100) - 1
        if last_sent_index == last_index - 1:
             last_sent_index = last_index
        sents = self._sents_by_range(tb_sents, first_sent_index, last_sent_index)
        # print "First index: %d / %d" % (first_sent_index, last_index)
        # print "Last index: %d / %d" % (last_sent_index, last_index)
        # print "%d sentences, Actual pct: %0.2f%%" % (len(sents), len(sents) / total_sents * 100)
        # print sents[0]
        # print sents[len(sents)-1]
        return sents
        
    def _sents_by_range(self, tb_sents, first_sent_index, last_sent_index):
        last_index = len(tb_sents) - 1
        if last_sent_index < first_sent_index:
            sents = tb_sents[first_sent_index:last_index+1] + tb_sents[0:last_sent_index+1]
        else:
            sents = tb_sents[first_sent_index:last_sent_index+1]
        return sents
        