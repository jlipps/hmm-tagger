from __future__ import division # use floating point division
from nltk import ConditionalFreqDist # for frequency distributions
from Helper import msg # for logging
from HMM import HMM # our Hidden Markov Model class
from Treebank import Treebank # our corpus class
import time

class Tagger:
    """
    A class for POS-tagging text and evaulating the result
    """
    
    ######### CLASS VARIABLES #########
    
    start_tag = '^' # a fake START tag to add to the beginning of sentences to help with bigram tagging
    default_tag = 'NN' # our fallback POS if we encounter an unknown word
    default_tag_upper = 'NNP' # our fallback POS if we encounter an unknown word
    
    def __init__(self, corpus_path, corpus_files):
        """
        Construct a Tagger object
        """
        
        self.tb = Treebank(corpus_path, corpus_files) # object for working with corpus data
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
        total_time_start = time.time()
        #self.tb.print_sents()
        train_pct = 90 # percentage of the corpus to train the tagger on
        test_pct = 10 # percentage of the corpus to test the tagger on
        rights = [] # array to hold number of correctly-tagged words for a given test
        wrongs = [] # array to hold number of incorrectly-tagged words for a given test
        totals = [] # array to hold 
        misseds = []
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
            (right, wrong, missed) = self.test(self.tb.testing_sents(test_pct,start_test_pct))
            total = right + wrong
            rights.append(right) # store the correct count for this test cycle
            wrongs.append(wrong) # store the incorrect count for this test cycle
            totals.append(total) # store the total words tested for this test cycle
            misseds = misseds + missed
            
            msg("Total words: %d\n" % total)
            msg("Correct tags: %d (%0.2f%%)\n" % (right, right / total * 100))
            msg("Incorrect tags: %d (%0.2f%%)\n" % (wrong, wrong / total * 100))
            
            if False:#raw_input("Examine bad tags? ") in ['y','Y']:
                self.inspect(missed)
            
        msg("%s%s" % (sep,sep))
        
        avg_right = sum(rights) / len(rights)
        avg_wrong = sum(wrongs) / len(wrongs)
        
        # output the results of our testing
        print "Total tests run: %d" % len(totals)
        print "Total time taken: %0.2f seconds" % (time.time() - total_time_start)
        print "Average correct tags: %0.2f%%" % (sum(rights) / sum(totals) * 100)
        print "Average incorrect tags: %0.2f%%" % (sum(wrongs) / sum(totals) * 100)
        print
        if raw_input("Examine bad tags? ") in ['y','Y']:
            self.inspect(misseds)
            
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
        self.words_given_pos_upper = ConditionalFreqDist((wp[1], wp[0]) for sent in sents for wp in sent)
        #self.words_given_pos = self._build_freq([(wp[1], wp[0].lower()) for sent in sents for wp in sent])
        # print set([wp[0].lower() for sent in sents for wp in sent if wp[1]=='IN'])
        # print set([wp[0].lower() for sent in sents for wp in sent if wp[1]=='WDT'])
        msg("done\n")
        
        # create another CFD that stores probabilities that stores observed
        # probabilities that one POS follows another POS
        msg("Training (Ci+1|Ci)...")
        self.pos2_given_pos1 = ConditionalFreqDist((sent[i-1][1], sent[i][1]) for sent in sents for i in range(1,len(sent)))
        #self.pos2_given_pos1 = self._build_freq([(sent[i-1][1], sent[i][1]) for sent in sents for i in range(1,len(sent))])
        msg("done\n")
        
    def _build_freq(self, condition_tuples):
        cfd_probs = {}
        conds = {}
        cfd_counts = {}
        for condition_tuple in condition_tuples:
            cond = condition_tuple[0]
            outcome = condition_tuple[1]
            if cond not in conds.keys():
                conds[cond] = 0
                cfd_counts[cond] = {}
                cfd_probs[cond] = {}
            if outcome not in cfd_counts[cond].keys():
                cfd_counts[cond][outcome] = 0
            conds[cond] += 1
            cfd_counts[cond][outcome] += 1
            cfd_probs[cond][outcome] = cfd_counts[cond][outcome] / conds[cond]
        
        return cfd_probs
        
    def test(self, sent_set):
        """
        Use a Hidden Markov Model to tag a set of sentences, and evaluate accuracy.
        
        :param sent_set: tuple like (untagged sentences, gold standard sentences)
        """
        
        untagged_sents = sent_set[0] # recover untagged sentences
        gold_tagged_sents = sent_set[1] # recover gold standard tagged sentences
        
        # initialize an HMM object with necessary parameters
        self.hmm = HMM(untagged_sents, self.pos_tags, self.words_given_pos, self.words_given_pos_upper, self.pos2_given_pos1, Tagger.default_tag, Tagger.default_tag_upper, Tagger.start_tag)
        
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
        
        # msg("Evaluating...")
        
        # ensure our sentence sets have the same length
        if len(hmm_tagged_sents) != len(gold_tagged_sents):
            raise Exception("HMM-tagged sentence set did not match gold standard sentence set!")
        
        right = 0 # initialize counter of correct tags
        wrong = 0 # initialize counter of incorrect tags
        missed = [] # initialize array of tagged words we didn't get
        
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
                    missed.append((hmm_tagged_word, gold_tagged_word, hmm_tagged_sents[i], gold_tagged_sents[i]))
                    wrong += 1
            # end words loop
        # end sentences loop
        
        #msg("done\n")
        
        # return a tuple of correct vs incorrect tags
        return (right, wrong, missed)
        
    def inspect(self, missed):
        cfd = ConditionalFreqDist((ms[1][1], ms[0][1]) for ms in missed)
        worst_mistakes = {}
        cfd.tabulate()
        msg("\n")
        conds = cfd.conditions()
        for g_tag in conds:
            for hmm_tag in cfd[g_tag].keys():
                count = cfd[g_tag][hmm_tag]
                if count not in worst_mistakes.keys():
                    worst_mistakes[count] = []
                worst_mistakes[count].append((hmm_tag, g_tag))
        mistake_counts = set([count for (count, mistake_set) in worst_mistakes.iteritems() if count > 50])
        mistake_counts = reversed(sorted(mistake_counts))
        mistakes_to_halt = []
        for count in mistake_counts:
            mistake_set = worst_mistakes[count]
            for mistake_tuple in mistake_set:
                mistakes_to_halt.append(mistake_tuple)
                msg("%d\t%s\twas really\t%s\n" % (count, mistake_tuple[0], mistake_tuple[1]))
        msg("\n")
        response = None
        for missed_set in missed:
            if response not in ['q','Q']:
                (hmm_tagged_word, gold_tagged_word, hmm_tagged_sent, gold_tagged_sent) = missed_set
                should_halt = False
                for pair in mistakes_to_halt:
                    if hmm_tagged_word[1] == pair[0] and gold_tagged_word[1] == pair[1]:
                        should_halt = True
                if should_halt:
                    msg("---------------------------------------------------\nTagged '%s' with %s when it should have been %s.\n-----------------------------------------\n" % (hmm_tagged_word[0], hmm_tagged_word[1], gold_tagged_word[1]))
                    msg("Gold: " + (' '.join([(w[0] + "/" + w[1]) for w in gold_tagged_sent])))
                    msg("\n-----------------------------------------\n")
                    msg("Mine: " + (' '.join([(w[0] + "/" + w[1]) for w in hmm_tagged_sent])))
                    response = raw_input("\n\nEnter to continue, Q to quit:")

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
        self.pos_tags.append('UNK')
        if '--' not in self.pos_tags:
            self.pos_tags.append('--')
        
        return new_sents