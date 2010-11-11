from __future__ import division # for floating-point division
from Helper import * # progress_bar(), indices_of_max(), msg()
from Guesser import Guesser
import time # for timing our tagging process
import re # for regex

class HMM:
    "A class for building Hidden Markov Models of tagged word data"
    punct_list = ["''", '``', ',']
    
    def __init__(self, untagged_sents, pos_tags, words_given_pos, words_given_pos_upper, pos2_given_pos1, default_tag, default_tag_upper, start_tag):
        """
        Construct a HMM object
        
        :param untagged_sents: list of untagged sentences for tagging
        :param pos_tags: list of possible POS tags
        :param words_given_pos: nltk.ConditionalFreqDist for P(Wi|Ck)
        :param pos2_given_pos1: nltk.ConditionalFreqDist for P(Ci+1|Ci)
        :param default_tag: POS tag to guess for words
        :param start_tag: start tag used to mark sentence beginning
        """
        
        self.default_tag = default_tag
        self.default_tag_upper = default_tag_upper
        self.start_tag = start_tag
        self.untagged_sents = untagged_sents
        self.all_pos_tags = pos_tags
        self.words_given_pos = words_given_pos
        self.words_given_pos_upper = words_given_pos_upper
        self.pos2_given_pos1 = pos2_given_pos1
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
        total = len(self.untagged_sents) # total sentences to tag
        total_prob_time = 0
        total_other_time = 0
        total_guess_count = 0
        total_word_count = 0
        total_unknown_count = 0
        
        for sent in self.untagged_sents:
            total_word_count += len(sent)
            (tagged_sent, prob_time, other_time, guess_count, unknown_count) = self.tag_sent(sent)
            total_prob_time += prob_time
            total_other_time += other_time
            total_guess_count += guess_count
            total_unknown_count += unknown_count
            tagged_sents.append(tagged_sent) # tag a sentence, add to array
            complete += 1 # increment our completed counter
            progress_bar(complete,total,time.time() - start_time) # show nice progress bar
        msg("\n")
        msg("Time spent looking up probabilities: %0.2fs\n" % total_prob_time)
        # msg("Total other time: %0.2f\n" % total_other_time)
        msg("Total unseen words: %d (%0.2f%% of total)\n" % (total_unknown_count, total_unknown_count / total_word_count * 100))
        msg("Total words guessed: %d (%0.2f%% of unseen)\n" % (total_guess_count, total_guess_count / total_unknown_count * 100))
        
        
        return tagged_sents
        
    def tag_sent(self, words):
        """
        Tag a sentence using the Viterbi algorithm
        
        :param words: a list of untagged words
        """
        
        prob_time = 0
        other_time = 0
        start_time = time.time()
        guess_count = 0
        unknown_count = 0
        words_range = range(len(words)) # reusable list: number of words in our sentence
        pos_range = range(len(self.all_pos_tags)) # reusable list: number of possible POS tags
        scores = [[None for j in words_range] for i in pos_range] # initialize i x j matrix
        backpointer = [[None for j in words_range] for i in pos_range] # initialize i x j matrix
        pos_tags = ['' for j in words_range] # initialize array of POS tags for this sentence
        pos_tag_indices = [None for j in words_range] # initialize array of POS tags for this sentence
        guessed_pos = [None for j in words_range] # initialize array of POS tags for this sentence
        
        cpwp = self._cp_of_word_given_pos # give helper function a shorthand name
        cpwpu = self._cp_of_upper_word_given_pos # give helper function a shorthand name
        cpp2p1 = self._cp_of_pos2_given_pos1 # give helper function a shorthand name
        
        # loop through words
        for j in words_range:
            word_j = words[j]
            is_upper = re.search(r'[A-Z]', word_j[0]) is not None
            scores_without_word_prob = [0 for i in pos_range]
            # loop through possible POS tags
            # posi_given_posks = [0 for i in pos_range]
            # wordj_given_posis = [0 for i in pos_range]                
            for i in pos_range:
                tag_i = self.all_pos_tags[i]
                # if this is the first word, perform initial calculation...
                if j==0:
                    cpwp_ji = cpwp(word_j.lower(), tag_i)
                    scores[i][j] = cpp2p1(tag_i, self.start_tag) * cpwp_ji
                    scores_without_word_prob[i] = cpp2p1(tag_i, self.start_tag)
                    backpointer[i][j] = 0
                else:

                    
                    # look only at those values of scores[x][j-1] (i.e., the score corresponding 
                    # to the likelihood that word[j-1] is pos[x]) that are the highest, 
                    # since we know we are going to multiply the new calculation with that.
                    # [scores[n][j-1] for n in pos_range] = the array of scores for word[j-1]
                    start_prob_time = time.time()
                    scores_pp2p1 = [-1 for m in pos_range]
                    for k in last_max_indices:
                        scores_pp2p1[k] = cpp2p1(tag_i, self.all_pos_tags[k])
                    max_pp2p1_score = max(scores_pp2p1)
                    max_k = scores_pp2p1.index(max_pp2p1_score)
                    if is_upper:# and i==self.all_pos_tags.index(self.default_tag_upper) and word_j is not "I" and word_j not in ['A','An','The','That']:
                        cpwp_ji = cpwpu(word_j, tag_i)
                    else:
                        if i==self.all_pos_tags.index(self.default_tag_upper):
                            cpwp_ji = 0
                        else:
                            cpwp_ji = cpwp(word_j, tag_i)
                    scores[i][j] = scores[max_k][j-1] * max_pp2p1_score * cpwp_ji
                    if words[j-1] in self.punct_list:
                        scores_without_word_prob[i] = 0
                    else:
                        scores_without_word_prob[i] = scores[max_k][j-1] * max_pp2p1_score
                    backpointer[i][j] = max_k
                    prob_time += time.time() - start_prob_time
            # end: for i in pos_range
            
            
            # take care that not all scores for this word are 0
            if self._smoothing_needed(scores, j_value=j):
                unknown_count += 1
                guess_tag = self.guesser.guess(word_j, scores_without_word_prob)
                if guess_tag == None:
                    guess_index=False
                else:
                    guess_index = self.all_pos_tags.index(guess_tag)
                    guess_count += 1
                (scores, did_guess) = self._smooth_values(scores, j_value=j, guess_index=guess_index)
            else:
                did_guess = False
            guessed_pos[j] = did_guess
            # if did_guess:
            #     print "Had to guess for %s, guessed %s" % (word_j, guess_tag)
            
            # get smoothed scores for this word
            scores_for_this_word = [scores[n][j] for n in pos_range]
 
            last_max_indices = indices_of_max(scores_for_this_word)
        
        # end: for j in words_range
        
        for j in reversed(words_range):
            col = [scores[i][j] for i in pos_range]
            if j==len(words_range)-1:
                pos_tag_indices[j] = col.index(max(col))
            else:
                pos_tag_indices[j] = backpointer[pos_tag_indices[j+1]][j+1]
        pos_tags = [self.all_pos_tags[index] for index in pos_tag_indices]
        
        # clean up obvious errors
        pos_tags = self.guesser.fix_tags(guessed_pos, pos_tags)
        
        # associate POS tags with words
        tagged_sent = [(words[j], pos_tags[j]) for j in words_range]
        
        end_time = time.time()
        other_time = end_time - start_time - prob_time
        
        return (tagged_sent, prob_time, other_time, guess_count, unknown_count)
        
    
    
    ######### `PRIVATE' FUNCTIONS #########
    
    def _compute_score(self, path_score, cpwp, cpp2p1):
        pass
        
    def _smooth2(self, array, guess_index=-1):
        if max(array)==0:
            if guess_index > -1:
                array[guess_index] = 1
            else:
                for i in range(len(array)):
                    array[i] = 1/len(array)
        return array
        
    def _cp_of_word_given_pos(self, word, pos):
        """
        Determine the conditional probability of a word given a POS
        
        :param word: string
        :param pos: string
        """
        
        # use the Conditional Frequency Distribution created by the trainer
        return self.words_given_pos[pos].freq(word)
        # try:
        #     cp = self.words_given_pos[pos][word]
        # except KeyError:
        #     cp = 0.0
        # return cp
        
    def _cp_of_upper_word_given_pos(self, word, pos):
        """
        Determine the conditional probability of a word given a POS

        :param word: string
        :param pos: string
        """

        # use the Conditional Frequency Distribution created by the trainer
        return self.words_given_pos_upper[pos].freq(word)
        # try:
        #     cp = self.words_given_pos[pos][word]
        # except KeyError:
        #     cp = 0.0
        # return cp
        
    def _cp_of_pos2_given_pos1(self, pos2, pos1):
        """
        Determine the conditional probability of one POS given another.
        This is, in other words, the probability that POS2 follows POS1.
        
        :param pos2: string
        :param pos1: string
        """
        
        # use the Conditional Frequency Distribution created by the trainer
        return self.pos2_given_pos1[pos1].freq(pos2)
        # try:
        #     cp = self.pos2_given_pos1[pos1][pos2]
        # except KeyError:
        #     cp = 0.0
        # return cp
        
    def _smoothing_needed(self, matrix, j_value):
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
        did_guess = False
        
        # assume we have all zeroes
        # if we want to prefer one row, give it a value of .75
        if guess_index > -1:
            guess_value = 0.75
            non_guess_value = (1 - guess_value) / len(matrix)
            for i in row_range:
                if i==guess_index:
                    matrix[guess_index][j_value] = guess_value
                else:
                    matrix[guess_index][j_value] = non_guess_value
            did_guess = True
            
        # otherwise, split the value of 1 over all rows
        else:
            for i in row_range:
                matrix[i][j_value] = 1 / len(matrix)

        return (matrix, did_guess)