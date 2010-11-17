######### Tagger.py #########

from __future__ import division # use floating point division
from nltk import ConditionalFreqDist # for frequency distributions
from Helper import msg # for logging
from HMM import HMM # our Hidden Markov Model class
from Treebank import Treebank # our corpus class
from PennTags import PennTags # our tag list
import time # for timing various processes

class Tagger:
    """
    A class for POS-tagging text and evaulating the result
    """
    
    ######### CLASS VARIABLES #########
    
    # a fake START tag to add to the beginning of sentences to help with tagging
    start_tag = '^'
    
    # number of times for a POS tagging mistake to occur in order to show it to user
    mistake_threshold = 50
    
    # x-fold cross-validation
    test_cycles = 10
    
    def __init__(self, corpus_path, corpus_files):
        """
        Construct a Tagger object
        
        :param corpus_path: path to corpus files
        :param corpus_files: list of corpus files
        """
        
        # object for working with corpus data
        self.tb = Treebank(corpus_path, corpus_files) 
        
        # will contain a list of tags in training corpus
        self.pos_tags = False 
        
        # will be object for running the Hidden Markov Model for tagging
        self.hmm = False
        
        # use PennTags
        self.tags = PennTags
        
        # will hold conditional frequency distribution for P(Wi|Ck)
        self.words_given_pos = False
        
        # will hold conditional frequency distribution for P(Ci+1|Ci) 
        self.pos2_given_pos1 = False
    
    
    ######### `PUBLIC' FUNCTIONS #########
    
    def run_test_cycles(self):
        """
        Run the test cycles for training and testing the tagger.
        Specifically, employ ten-fold cross-validation to train/test on different
        segments of the corpus.
        """
        
        total_time_start = time.time() # keep track of time
        pct_step = int(100 / Tagger.test_cycles) # cycle steps in pct
        test_pct = pct_step # percentage of the corpus to test the tagger on
        train_pct = 100 - test_pct # percentage of the corpus to train the tagger on
        rights = [] # array to hold number of correctly-tagged words for each test
        wrongs = [] # array to hold number of incorrectly-tagged words for each test
        totals = [] # array to hold number of total words for each test
        all_missed = [] # array to hold incorrect tag information for each test
        sep = ''.join(["-" for i in range(50)]) + "\n" # logging separator
        
        # loop from 0-90 (step size 10)
        for start_train_pct in [x*pct_step for x in range(Tagger.test_cycles)]:
            msg("%sSTARTING TEST CYCLE %d\n%s" % (sep, (start_train_pct/pct_step)+1,\
                sep))
            
            # find the percent point to start collecting test sentences
            # may be > 100, so circle round
            start_test_pct = (start_train_pct+train_pct) % 100
            
            # train the tagger on sentences from the corpus matching our range
            training_sents = self.tb.training_sents(train_pct,start_train_pct)
            self.train(training_sents)
            
            # test the tagger on the rest of the sentences
            testing_sents = self.tb.testing_sents(test_pct,start_test_pct)
            (right, wrong, missed) = self.test(testing_sents)
            
            # gather accuracy statistics for this test
            total = right + wrong
            rights.append(right) # store the correct count for this test cycle
            wrongs.append(wrong) # store the incorrect count for this test cycle
            totals.append(total) # store the total words tested for this test cycle
            all_missed += missed # add incorrect tag information from this cycle
            
            msg("Total words: %d\n" % total)
            msg("Correct tags: %d (%0.2f%%)\n" % (right, right / total * 100))
            msg("Incorrect tags: %d (%0.2f%%)\n" % (wrong, wrong / total * 100))
        # end: test cycle
            
        msg("%s%s" % (sep,sep))
        
        # calculate and output statistics for the entire test
        print "Total tests run: %d" % len(totals)
        print "Total time taken: %0.2f seconds" % (time.time() - total_time_start)
        print "Average correct tags: %0.2f%%" % (sum(rights) / sum(totals) * 100)
        print "Average incorrect tags: %0.2f%%" % (sum(wrongs) / sum(totals) * 100)
        print
        
        # give the option of inspecting incorrect tags
        if raw_input("Examine bad tags? ") in ['y','Y']:
            self.inspect(all_missed)
            
    def train(self, sents):
        """
        Train the tagger on a set of tagged sentences
        
        :param sents: list of tagged sentences
        """
        
        # collect POS tags from our corpus
        self.pos_tags = self.tb.pos_tags()
        
        # add start markers to help with bigram tagging
        msg("Adjusting POS tags...")
        sents = self._adjust_pos(sents)
        msg("done\n")
        
        # create 2 conditional frequency distributions (from the NLTK) that store
        # observed probabilities that a given word has a certain POS, one for
        # lowercase-normalized words and one for words as they appear in the text
        msg("Training (Wi|Ck)...")
        
        # create a CFD for words normalized to lowercase
        self.words_given_pos = ConditionalFreqDist((wp[1], wp[0].lower()) for \
            sent in sents for wp in sent)
            
        # create a CFD for words left in their original capitalization
        self.words_given_pos_upper = ConditionalFreqDist((wp[1], wp[0]) for \
            sent in sents for wp in sent)
        msg("done\n")
        
        # create another CFD that stores probabilities that stores observed
        # probabilities that one POS follows another POS
        msg("Training (Ci+1|Ci)...")
        self.pos2_given_pos1 = ConditionalFreqDist((sent[i-1][1], sent[i][1]) for \
            sent in sents for i in range(1,len(sent)))

        msg("done\n")
        
    def test(self, sent_set):
        """
        Use a Hidden Markov Model to tag a set of sentences, and evaluate accuracy.
        
        :param sent_set: tuple like (untagged sentences, gold standard sentences)
        """
        
        untagged_sents = sent_set[0] # recover untagged sentences
        gold_tagged_sents = sent_set[1] # recover gold standard tagged sentences
        
        # initialize an HMM object with necessary parameters
        self.hmm = HMM(untagged_sents, self.pos_tags, self.words_given_pos, \
            self.words_given_pos_upper, self.pos2_given_pos1, Tagger.start_tag)
        
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
        
        # ensure our sentence sets have the same length
        if len(hmm_tagged_sents) != len(gold_tagged_sents):
            raise Exception("HMM-tagged sentence set did not match gold \
                standard sentence set!")
        
        right = 0 # initialize counter of correct tags
        wrong = 0 # initialize counter of incorrect tags
        missed = [] # initialize array of tagged words we didn't get right
        
        # loop through sentence sets
        for i in range(len(gold_tagged_sents)):
            
            # ensure our sentences have the same length
            if len(hmm_tagged_sents[i]) != len(gold_tagged_sents[i]):
                raise Exception("HMM-tagged sentence did not match gold \
                    standard sentence!")
                
            # loop through words in sentence
            for j in range(len(gold_tagged_sents[i])):
                gold_tagged_word = gold_tagged_sents[i][j]
                hmm_tagged_word = hmm_tagged_sents[i][j]
                
                # ensure the words are the same between the sets
                if gold_tagged_word[0] != hmm_tagged_word[0]:
                    raise Exception("HMM-tagged word did not match gold \
                        standard word!")

                # increment counters based on tag correctness
                if gold_tagged_word[1] == hmm_tagged_word[1]:
                    right += 1
                else:
                    missed.append((hmm_tagged_word, gold_tagged_word, \
                        hmm_tagged_sents[i], gold_tagged_sents[i]))
                    wrong += 1
            # end words loop
        # end sentences loop
        
        # return a tuple of correct vs incorrect tags
        return (right, wrong, missed)
        
    def inspect(self, missed):
        """
        Inspect a testing session, and print data about tag accuracy
        
        :param missed: list of tuples of missed tags like:
            (hmm_tagged_word, gold_tagged_word, hmm_context, gold_context)
        """
        
        # create a CFD so we can examine a matrix of incorrect vs correct tags
        # ms[1][1] = tag of a gold_tagged_word
        # ms[0][1] = tag of an hmm_tagged_word
        cfd = ConditionalFreqDist((ms[1][1], ms[0][1]) for ms in missed)
        
        # initialize a hash to store mistakes by frequency
        mistakes = {}
        
        # print a table showing mistake frequency
        cfd.tabulate()
        msg("\n")
        
        # loop through mistake frequencies by gold standard tag, i.e., if we are
        # examining gold-standard 'IN', count what we incorrectly tagged it as
        conds = cfd.conditions()
        for g_tag in conds:
            for hmm_tag in cfd[g_tag].keys():
                # how many times did we incorrectly say g_tag was hmm_tag?
                count = cfd[g_tag][hmm_tag]
                
                # add these mistakes to the count
                if count not in mistakes.keys():
                    mistakes[count] = []
                mistakes[count].append((hmm_tag, g_tag))
                
        # get a list of all mistake types that occurred over a threshold, worst first
        mistake_counts = set([count for (count, mistake_set) in \
            mistakes.iteritems() if count > Tagger.mistake_threshold])
        mistake_counts = reversed(sorted(mistake_counts))
        
        # now create a list of mistake types to show the user, i.e., loop 
        # through all types and if they are of a high-frequency type, add to list
        mistakes_to_halt = []
        for count in mistake_counts:
            mistake_set = mistakes[count]
            for mistake_tuple in mistake_set:
                mistakes_to_halt.append(mistake_tuple)
                msg("%d\t%s\twas really\t%s\n" % (count, mistake_tuple[0], \
                    mistake_tuple[1]))
        msg("\n")
        
        # create separators used when outputting missed word contexts
        sep_big = "---------------------------------------------------\n"
        sep_small = "\n-----------------------------------------\n"
        
        # loop through individual mistakes and, if they match the kind of error
        # we want to halt for, show the user the mistake as well as the sentence
        # context for both the gold-standard sentence and the hmm-tagged sentence
        response = None
        for missed_set in missed:
            if response not in ['q','Q']:
                (hmm_tagged_word, gold_tagged_word, hmm_tagged_sent, \
                    gold_tagged_sent) = missed_set
                should_halt = False
                # determine whether the current mistake matches a mistake type
                # we want to halt for
                for pair in mistakes_to_halt:
                    if hmm_tagged_word[1] == pair[0] and \
                        gold_tagged_word[1] == pair[1]:
                        should_halt = True
                if should_halt:
                    msg("%sTagged '%s' with %s when it should have been %s.%s" %\
                    (sep_big, hmm_tagged_word[0], hmm_tagged_word[1],\
                        gold_tagged_word[1], sep_small))
                    
                    msg("Gold: " + (' '.join([(w[0] + "/" + w[1]) for w in \
                        gold_tagged_sent])))
                    msg(sep_small)
                    msg("Mine: " + (' '.join([(w[0] + "/" + w[1]) for w in \
                        hmm_tagged_sent])))
                    
                    # get user input to decide whether to keep going
                    response = raw_input("\n\nEnter to continue, Q to quit: ")

    ######### `PRIVATE' FUNCTIONS #########
    
    def _adjust_pos(self, sents):
        """
        Insert start markers (word and tag tuple) in each sentence of a list.
        Add any other tags that need adding
        
        :param sents: list of tagged sentences
        """
        
        new_sents = [] # initialize array of start-marked sentences
        
        # loop through tagged sentences
        for sent in sents:
            # add a new start-marked sentence to our array
            new_sents.append([(Tagger.start_tag, Tagger.start_tag)] + sent)
            
        # make sure our start marker tag gets added to the POS list
        self.pos_tags.append(Tagger.start_tag)
        
        # also take the opportunity to add other tags to the list
        # which we may not have encountered in testing
        for tag in self.tags.rare_tags:
            if tag not in self.pos_tags:
                self.pos_tags.append(tag)
        
        return new_sents