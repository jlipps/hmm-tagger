from __future__ import division # use floating point division
from nltk import ConditionalFreqDist # for frequency distributions
from Helper import msg # for logging
from HMM import HMM # our Hidden Markov Model class
from Treebank import Treebank # our corpus class

class Tagger:
    """
    A class for POS-tagging text and evaulating the result
    """
    
    ######### CLASS VARIABLES #########
    
    start_tag = '^' # a fake START tag to add to the beginning of sentences to help with bigram tagging
    default_tag = 'NNP' # our fallback POS if we encounter an unknown word
    
    def __init__(self):
        """
        Construct a Tagger object
        """
        
        self.tb = Treebank() # object for working with corpus data
        self.pos_tags = False # will contain a list of tags in training corpus
        self.hmm = False # object for running the Hidden Markov Model for tagging
        self.words_given_pos = False # conditional frequency distribution for P(Wi|Cj)
        self.pos2_given_pos1 = False # conditional frequency distribution for P(Ci+1|Ci)
    
    
    ######### `PUBLIC' FUNCTIONS #########
    
    def run_test_cycles(self):
        """
        Run the test cycles for training and testing the tagger.
        Specifically, employ ten-fold cross-validation to train/test on different
        segments of the corpus.
        """
        train_pct = 90 # percentage of the corpus to train the tagger on
        test_pct = 10 # percentage of the corpus to test the tagger on
        rights = [] # array to hold number of correctly-tagged words for a given test
        wrongs = [] # array to hold number of incorrectly-tagged words for a given test
        totals = [] # array to hold 
        sep = "-----------------------------------------------\n" # logging separator
        
        # loop from 0-90 (step size 10)
        for start_train_pct in [x*10 for x in range(10)]:
            msg("%sSTARTING TEST CYCLE %d\n%s" % (sep, (start_train_pct/10)+1, sep))
            
            # find the percent point to start collecting test sentences
            # may be > 100, so circle round
            start_test_pct = (start_train_pct+train_pct) % 100
            
            # train the tagger on sentences from the corpus matching our range
            self.train(self.tb.training_sents(train_pct,start_train_pct))
            
            # test the tagger on the rest of the sentences, getting an accuracy measure
            (right, wrong) = self.test(self.tb.testing_sents(test_pct,start_test_pct))
            total = right + wrong
            rights.append(right) # store the correct count for this test cycle
            wrongs.append(wrong) # store the incorrect count for this test cycle
            totals.append(total) # store the total words tested for this test cycle
            
            msg("Total words: %d\n" % total)
            msg("Correct tags: %d (%0.2f%%)\n" % (right, right / total * 100))
            msg("Incorrect tags: %d (%0.2f%%)\n" % (wrong, wrong / total * 100))
            
        msg("%s%s" % (sep,sep))
        
        avg_right = sum(rights) / len(rights)
        avg_wrong = sum(wrongs) / len(wrongs)
        
        # output the results of our testing
        print "Total tests run: %d" % len(rights)
        print "Average correct tags: %d (%0.2f%%)" % (avg_right, sum(rights) / sum(totals) * 100)
        print "Average incorrect tags: %d (%0.2f%%)" % (avg_wrong, sum(wrongs) / sum(totals) * 100)
        print
            
    def train(self, sents):
        """
        Train the tagger on a set of tagged sentences
        
        :param sents: list of tagged sentences
        """
        
        # collect POS tags from our corpus
        self.pos_tags = self.tb.pos_tags()
        
        # add start markers to help with bigram tagging
        msg("Inserting start markers...")
        sents = self._insert_start_markers(sents)
        msg("done\n")
        
        # create a conditional frequency distribution (from the NLTK) that stores
        # observed probabilities that a given word has a certain POS
        msg("Training (Wi|Ck)...")
        self.words_given_pos = ConditionalFreqDist((wp[1], wp[0].lower()) for sent in sents for wp in sent)
        msg("done\n")
        
        # create another CFD that stores probabilities that stores observed
        # probabilities that one POS follows another POS
        msg("Training (Ci+1|Ci)...")
        self.pos2_given_pos1 = ConditionalFreqDist((sent[i-1][1], sent[i][1]) for sent in sents for i in range(1,len(sent)))
        msg("done\n")
        
    def test(self, sent_set):
        """
        Use a Hidden Markov Model to tag a set of sentences, and evaluate accuracy.
        
        :param sent_set: tuple like (untagged sentences, gold standard sentences)
        """
        
        untagged_sents = sent_set[0] # recover untagged sentences
        gold_tagged_sents = sent_set[1] # recover gold standard tagged sentences
        
        # initialize an HMM object with necessary parameters
        self.hmm = HMM(untagged_sents, self.pos_tags, self.words_given_pos, self.pos2_given_pos1, Tagger.default_tag, Tagger.start_tag)
        
        # get HMM-tagged sentences
        hmm_tagged_sents = self.hmm.tag()
        
        # evaluate against gold standard and return accuracy data
        return self.evaluate(hmm_tagged_sents, gold_tagged_sents)
        
    def evaluate(self, hmm_tagged_sents, gold_tagged_sents):
        """
        Evaluate one set of tagged sentences against another set
        
        :param hmm_tagged_sents: list of tagged sentences
        :param gold_tagged_sents: list of tagged sentences used as gold standard
        """
        
        msg("Evaluating...")
        
        # ensure our sentence sets have the same length
        if len(hmm_tagged_sents) != len(gold_tagged_sents):
            raise Exception("HMM-tagged sentence set did not match gold standard sentence set!")
        
        right = 0 # initialize counter of correct tags
        wrong = 0 # initialize counter of incorrect tags
        
        # loop through sentence sets
        for i in range(len(gold_tagged_sents)):
            
            # ensure our sentences have the same length
            if len(hmm_tagged_sents[i]) != len(gold_tagged_sents[i]):
                raise Exception("HMM-tagged sentence did not match gold standard sentence!")
                
            # loop through words in sentence
            for j in range(len(gold_tagged_sents[i])):
                gold_tagged_word = gold_tagged_sents[i][j]
                hmm_tagged_word = hmm_tagged_sents[i][j]
                
                # ensure the words are the same between the sets
                if gold_tagged_word[0] != hmm_tagged_word[0]:
                    raise Exception("HMM-tagged word did not match gold standard word!")

                # increment counters based on tag correctness
                if gold_tagged_word[1] == hmm_tagged_word[1]:
                    right += 1
                else:
                    wrong += 1
            # end words loop
        # end sentences loop
        
        msg("done\n")
        
        # return a tuple of correct vs incorrect tags
        return (right, wrong)

    ######### `PRIVATE' FUNCTIONS #########
    
    def _insert_start_markers(self, sents):
        """
        Insert start markers (word and tag tuple) in each sentence of a list.
        
        :param sents: list of tagged sentences
        """
        
        new_sents = [] # initialize array of start-marked sentences
        
        # loop through tagged sentences
        for sent in sents:
            # add a new start-marked sentence to our array
            new_sents.append([(Tagger.start_tag, Tagger.start_tag)] + sent)
            
        # make sure our start marker tag gets added to the POS list
        self.pos_tags.append(Tagger.start_tag)
        
        return new_sents