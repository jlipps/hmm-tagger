######### HMM.py #########

from __future__ import division # for floating-point division
from Helper import * # for progress_bar(), indices_of_max(), msg()
from Guesser import Guesser # for word guesser
import time # for timing our tagging process
import re # for regex

class HMM:
    "A class for building Hidden Markov Models of tagged word data"
    
    ######### CLASS VARIABLES #########
    
    # store a small list of punctuation to help with training P(Ci+1|Ci)
    punct_list = ["''", '``', ',']
    
    def __init__(self, untagged_sents, pos_tags, words_given_pos, \
        words_given_pos_upper, pos2_given_pos1, start_tag):
        """
        Construct a HMM object
        
        :param untagged_sents: list of untagged sentences for tagging
        :param pos_tags: list of possible POS tags
        :param words_given_pos: nltk.ConditionalFreqDist for P(Wi|Ck) with all words
            converted to lowercase
        :param words_given_pos_upper: nltk.ConditionalFreqDist for P(Wi|Ck) with
            words left in original capitalization
        :param pos2_given_pos1: nltk.ConditionalFreqDist for P(Ci+1|Ci)
        :param default_tag: POS tag to guess for words
        :param start_tag: start tag used to mark sentence beginning
        """
        
        self.start_tag = start_tag
        self.untagged_sents = untagged_sents
        self.num_untagged_sents = len(untagged_sents)
        self.all_pos_tags = pos_tags
        self.words_given_pos = words_given_pos
        self.words_given_pos_upper = words_given_pos_upper
        self.pos2_given_pos1 = pos2_given_pos1
        
        # initialize one guesser object to use for the whole test
        self.guesser = Guesser(pos_tags, words_given_pos)
    
    ######### `PUBLIC' FUNCTIONS #########
        
    def tag(self):
        """
        Tag all this object's sentences, return a list of tagged sentences
        """
        
        msg("Tagging sentences:\n")
        start_time = time.time() # mark the start time for this process
        tagged_sents = [] # array to hold tagged sentences
        complete = 0 # how many sentences we have tagged
        
        # initialize variables to track for tagging stats
        total_prob_time = 0 # time spent looking up probabilities
        total_other_time = 0 # time spent doing other things
        total_guess_count = 0 # words we used the guesser to guess POS for
        total_word_count = 0 # num words tagged
        total_unknown_count = 0 # num words with no P(Wi|Ci)
        
        # tag each sentence and track statistics
        for sent in self.untagged_sents:
            total_word_count += len(sent)
            (tagged_sent, prob_time, other_time, guess_count, unknown_count) = \
                self.tag_sent(sent)
            total_prob_time += prob_time
            total_other_time += other_time
            total_guess_count += guess_count
            total_unknown_count += unknown_count
            tagged_sents.append(tagged_sent) # append tagged sentence to array
            complete += 1 # increment our completed counter for progress bar
            # show nice progress bar
            progress_bar(complete,self.num_untagged_sents,time.time() - start_time)
            
        # print nice things to the user
        msg("\n")
        msg("Time spent looking up probabilities: %0.2fs\n" % total_prob_time)
        msg("Total unseen words: %d (%0.2f%% of total)\n" % (total_unknown_count, \
            total_unknown_count / total_word_count * 100))
        msg("Total words guessed: %d (%0.2f%% of unseen)\n" % (total_guess_count, \
            total_guess_count / total_unknown_count * 100))
        
        
        return tagged_sents
        
    def tag_sent(self, words):
        """
        Tag a sentence using the Viterbi algorithm
        
        :param words: a list of untagged words
        """
        
        # initialize stats tracking variables
        prob_time = 0
        other_time = 0
        start_time = time.time()
        guess_count = 0
        unknown_count = 0
        
        # initialize arrays used for algorithm
        
        # reusable looping list: number of words in our sentence
        words_range = range(len(words))
        
        # reusable looping list: number of possible POS tags
        pos_range = range(len(self.all_pos_tags))
        
        # initialize i x j matrix to hold scores
        scores = [[None for j in words_range] for i in pos_range]
        
        # initialize i x j matrix to hold backpointers
        backpointer = [[None for j in words_range] for i in pos_range]
        
        # initialize array of POS tags for this sentence
        pos_tags = ['' for j in words_range]
        
        # initialize array of POS tags for words in sentence
        pos_tag_indices = [None for j in words_range]
        
        # initialize array of guess states for words in sentence
        guessed_pos = [None for j in words_range]
        
        # initialize count of words we guessed on for reporting
        guess_count = 0
        
        # give P(Wi|Ck) trained with lowercase a shorthand name
        cpwp = lambda word,pos: self.words_given_pos[pos].freq(word)
        
        # give P(Wi|Ck) trained with normal capitalization a shorthand name
        cpwpu = lambda word,pos: self.words_given_pos_upper[pos].freq(word)
        
        # give P(Ci+1|Ci)   a shorthand name
        cpp2p1 = lambda pos2,pos1: self.pos2_given_pos1[pos1].freq(pos2)
        
        # loop through words
        for j in words_range:
            word_j = words[j] # store current word in a local variable
            
            # determine whether word begins with a capital letter
            is_upper = re.search(r'[A-Z]', word_j[0]) is not None
            
            # initialize an array to hold the scores for this word not taking into
            # account the word probability, i.e., including only the path and
            # the bare POS probability
            scores_without_word_prob = [0 for i in pos_range]
            
            # loop through possible POS tags
            for i in pos_range:
                tag_i = self.all_pos_tags[i] # POS tag for this POS index
                
                # if this is the first word, perform initial calculation...
                if j==0:
                    # find P(Wj|Ci) using lowercase since in the first word,
                    # capitalization is not helpful information
                    cpwp_ji = cpwp(word_j.lower(), tag_i)
                    
                    # find P(Ci|'^')
                    cp_istart = cpp2p1(tag_i, self.start_tag)
                    
                    # calculate score using P(Ci|'^') and P(Wj|Ci)
                    scores[i][j] = cp_istart * cpwp_ji
                    
                    # also find bare POS probability, in this case the same as
                    # P(Ci|'^')
                    scores_without_word_prob[i] = cp_istart
                    
                    # initialize backpointer for this word to 0
                    backpointer[i][j] = 0
                    
                # if we're not looking at the first word...
                else:
                    start_prob_time = time.time() # start our prob lookup timer
                    
                    # initialize an array corresponding to all the POS tags with 1
                    # in each slot. This will hold the probability that POS i is
                    # what it is given that it may have followed any other POS
                    scores_pp2p1 = [-1 for m in pos_range]
                    
                    # we don't actually need to lookup this conditional probability
                    # for every POS, since we know which POS for words[j-1] have the
                    # highest score so far. Thus we only look at those POS in 
                    # last_max_indices, which stores the POS indices of the POS that
                    # scored highest for word[j-1]
                    for k in last_max_indices:
                        scores_pp2p1[k] = cpp2p1(tag_i, self.all_pos_tags[k])
                        
                    # now we want to find the highest P(Ci|Ck) score
                    max_pp2p1_score = max(scores_pp2p1)
                    
                    # also, get the POS index (k from Ck) corresponding to it
                    max_k = scores_pp2p1.index(max_pp2p1_score)
                    
                    # now we find P(Wj|Ci)
                    if is_upper:
                        # if Wj is uppercase, look in the uppercase freq table
                        cpwp_ji = cpwpu(word_j, tag_i)
                    else:
                        # if Wj is lowercase, we know first of all that it can't be
                        # a proper noun, so remove these from the running
                        if tag_i in [self.guesser.tags.proper_noun, \
                            self.guesser.tags.pl_proper_noun]:
                            cpwp_ji = 0
                            
                        # otherwise, lookup the probability from the lowercase
                        # freq table
                        else:
                            cpwp_ji = cpwp(word_j, tag_i)
                            
                    # calculate the score for this word and possible POS as (a) the
                    # best score from the path so far, (b) the best possible score
                    # for the POS under consideration, and (c) P(Wj|Ci)
                    scores[i][j] = scores[max_k][j-1] * max_pp2p1_score * cpwp_ji

                    # keep track of the score for this POS without taking into 
                    # account P(Wj|Ci), so if word_j is an untrained word, we can
                    # use bare POS frequencies to help
                    scores_without_word_prob[i] = scores[max_k][j-1] * \
                        max_pp2p1_score
                    
                    # assert that the path to this word/POS combo came through the
                    # POS which gave us the highest score in our calculation,
                    # so we can recover the best POS for each word at the end
                    backpointer[i][j] = max_k
                    
                    prob_time += time.time() - start_prob_time
            # end: for i in pos_range
            
            did_guess = False
            # take care that not all scores for this word are 0
            if self._smoothing_needed(scores, j_value=j):
                # if all the scores are zero, guess that we've never seen this word
                # in training
                unknown_count += 1
                
                # try to guess a tag for this word based on its form and the bare
                # POS scores (i.e., guess based on form and then based on the
                # previous POS)
                guess_tag = self.guesser.guess(word_j, scores_without_word_prob)
                
                # if we didn't come up with a guess, make sure our smoother doesn't
                # weight any POS over any other
                if guess_tag == None:
                    guess_index=False
                    
                # otherwise, tell our smoother that we have a guess so that it
                # weights the guessed POS highest
                else:
                    # determine the index of the guessed POS tag
                    guess_index = self.all_pos_tags.index(guess_tag)
                    did_guess = True
                    guess_count += 1
                    
                # get a smoothed column of scores for scores[j]
                scores = self._smooth_values(scores, j_value=j, \
                    guess_index=guess_index)

            # record whether or not we guessed the POS for this word
            guessed_pos[j] = did_guess
            
            # turn the score column into a 1-dimensional array so we can more easily
            # find the best POS for this word
            scores_for_this_word = [scores[n][j] for n in pos_range]
 
            # get the POS indices which performed best for this word to pass on to
            # the algorithm for the next word, so it can only compute scores for
            # realistically likely POS
            last_max_indices = indices_of_max(scores_for_this_word)
        
        # end: for j in words_range
        
        # recover the POS tag indices for words in the sentence that led to the best
        # final scores
        for j in reversed(words_range):
            # get the column representing scores for each POS possible for words[j]
            col = [scores[i][j] for i in pos_range]
            
            # our last POS is whichever had the highest score in the last column
            if j==len(words_range)-1:
                pos_tag_indices[j] = col.index(max(col))
                
            # otherwise the POS is whichever the backpointer pointed to from the
            # next word
            else:
                pos_tag_indices[j] = backpointer[pos_tag_indices[j+1]][j+1]
                
        # get the actual tags for the indices recovered
        pos_tags = [self.all_pos_tags[index] for index in pos_tag_indices]
        
        # associate POS tags with words
        tagged_sent = [(words[j], pos_tags[j]) for j in words_range]
        
        # calculate time stats
        end_time = time.time()
        other_time = end_time - start_time - prob_time
        
        # return a bundle of tag data and other stats
        return (tagged_sent, prob_time, other_time, guess_count, unknown_count)
        
    
    
    ######### `PRIVATE' FUNCTIONS #########
        
    def _smoothing_needed(self, matrix, j_value):
        """
        Determine whether smoothing is needed for a column of a matrix
        
        :param matrix: the list of lists to examine
        :param j_value: the index of the column to examine for smoothing, i.e.,
            matrix[j]
        """
        return max([matrix[i][j_value] for i in range(len(matrix))]) == 0
        
    def _smooth_values(self, matrix, j_value=0, guess_index=-1):
        """
        Ensure that a column of a matrix is not full of zeroes.
        
        :param matrix: list of lists of numbers
        :param j_value: matrix column (default: 0)
        :param guess_index: row index to prefer (default: -1)
        """
        
        row_range = range(len(matrix)) # range for looping through rows
        
        # fill an array with values from the column
        value_array = [matrix[i][j_value] for i in row_range]
        
        # assume we have all zeroes
        # if we want to prefer one row, give it a certain value 0 - 1
        if guess_index > -1:
            guess_value = 0.75
            # give everything else the remaining probability distributed evenly
            non_guess_value = (1 - guess_value) / len(matrix)
            for i in row_range:
                # give our preferred row the weighted value
                if i==guess_index:
                    matrix[guess_index][j_value] = guess_value
                    
                # give every other row the rest of the probability distribution
                else:
                    matrix[guess_index][j_value] = non_guess_value
            
        # otherwise, simply split the probability value of 1 evenly over all rows
        else:
            for i in row_range:
                matrix[i][j_value] = 1 / len(matrix)

        return matrix