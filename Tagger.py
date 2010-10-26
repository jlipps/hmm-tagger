import nltk
from nltk import FreqDist, ConditionalFreqDist
from HMM import HMM
from Treebank import Treebank

class Tagger:
    "A class for POS-tagging data"
    
    def __init__(self):
        self.tb = Treebank()
        self.pos_tags = False
        self.hmm = False
        self.words_given_pos = False
        self.pos2_given_pos1 = False
        
    def run_test_cycle(self):
        train_pct = 90
        test_pct = 10
        for i in [x*10 for x in range(1)]:
            start_test_pct = (i+train_pct) % 100
            self.train(self.tb.training_sents(train_pct,i))
            self.test(self.tb.testing_sents(test_pct=test_pct,start_test_pct=start_test_pct))
            
    def train(self, sents):
        self.pos_tags = self.tb.pos_tags()
        sents = self._insert_start_markers(sents)
        self.words_given_pos = ConditionalFreqDist((wp[1], wp[0]) for sent in sents for wp in sent)
        #print self.words_given_pos['^'].freq('^')
        self.pos2_given_pos1 = ConditionalFreqDist((sent[i][1], sent[i-1][1]) for sent in sents for i in range(len(sent)-1) if i > 0)
        #print self.pos2_given_pos1['NN'].freq('JJ')
        
    def test(self, sent_set):
        untagged_sents = sent_set[0]
        gold_tagged_sents = sent_set[1]
        print len(self.pos_tags)
        self.hmm = HMM(untagged_sents, self.pos_tags, self.words_given_pos, self.pos2_given_pos1)
        hmm_tagged_sents = self.hmm.tag()
        self.evaluate(hmm_tagged_sents, gold_tagged_sents)
        
    def evaluate(self, hmm_tagged_sents, gold_tagged_sents):
        pass
        
    # `PRIVATE' FUNCTIONS
    
    
    
    def _insert_start_markers(self, sents):
        new_sents = []
        for sent in sents:
            new_sents.append([('^', '^')] + sent)
        self.pos_tags.append('^')
        return new_sents