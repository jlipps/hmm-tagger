from __future__ import division # for floating-point division
from Helper import * # progress_bar(), indices_of_max(), msg()
import time # for timing our tagging process

class HMM:
    "A class for building Hidden Markov Models of tagged word data"
    
    def __init__(self, untagged_sents, pos_tags, words_given_pos, pos2_given_pos1, default_tag, start_tag):
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
        self.start_tag = start_tag
        self.untagged_sents = untagged_sents
        self.all_pos_tags = pos_tags
        self.words_given_pos = words_given_pos
        self.pos2_given_pos1 = pos2_given_pos1
    
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
        
        for sent in self.untagged_sents:
            tagged_sents.append(self.tag_sent(sent)) # tag a sentence, add to array
            complete += 1 # increment our completed counter
            progress_bar(complete,total,time.time() - start_time) # show nice progress bar
        msg("\n")
        
        return tagged_sents
        
    def tag_sent(self, words):
        """
        Tag a sentence using the Viterbi algorithm
        
        :param words: a list of untagged words
        """
        
        words_range = range(len(words)) # reusable list: number of words in our sentence
        pos_range = range(len(self.all_pos_tags)) # reusable list: number of possible POS tags
        scores = [[None for j in words_range] for i in pos_range] # initialize i x j matrix
        pos_tags = ['' for j in words_range] # initialize array of POS tags for this sentence
        
        cpwp = self._cp_of_word_given_pos # give helper function a shorthand name
        cpp2p1 = self._cp_of_pos2_given_pos1 # give helper function a shorthand name
        
        # loop through words
        for j in words_range:
            
            # loop through possible POS tags
            for i in pos_range:
                
                # if this is the first word, perform initial calculation...
                if j==0:
                    # the score for each possible POS Ci for the first word is:
                    # P(W0|Ci) * P(Ci|'^'), where '^' is whatever start tag we use
                    scores[i][j] = cpwp(words[0].lower(), self.all_pos_tags[i]) * cpp2p1(self.all_pos_tags[i], self.start_tag)
                    
                # but in all other cases, make use of word[j-1]
                else:
                    
                    # create an array to keep track of possible paths to this combination
                    # of word[j] and pos[i]
                    scores_for_this_path = [0 for k in pos_range]
                    
                    # look only at those values of scores[x][j-1] (i.e., the score corresponding 
                    # to the likelihood that word[j-1] is pos[x]) that are the highest, 
                    # since we know we are going to multiply the new calculation with that.
                    # [scores[n][j-1] for n in pos_range] = the array of scores for word[j-1]
                    for k in indices_of_max([scores[n][j-1] for n in pos_range]):
                        
                        # calculate P(Ci|Ck)
                        posi_given_posk = cpp2p1(self.all_pos_tags[i], self.all_pos_tags[k])
                        
                        # calculate P(Wj|Ci)
                        wordj_given_posi = cpwp(words[j].lower(), self.all_pos_tags[i])
                        
                        # calculate a score for this path based on high score for last word
                        scores_for_this_path[k] = scores[k][j-1] * posi_given_posk * wordj_given_posi
                        
                    # set the score for this word/POS combo to the highest score
                    scores[i][j] = max(scores_for_this_path)
                # end: if j == 0
            
            # take care that not all scores for this word are 0
            scores = self._smooth_values(scores, j_value=j)
            
            # get smoothed scores for this word
            scores_for_this_word = [scores[n][j] for n in pos_range]
            
            # if all POS tags have the same score for this word
            if scores_for_this_word.count(max(scores_for_this_word)) == len(scores_for_this_word):
                # guess the default tag
                pos_for_this_word = self.default_tag
                
            # if not all POS tags have the same score for this word
            else:
                # pick the first POS with the high score for this word
                pos_for_this_word = self.all_pos_tags[scores_for_this_word.index(max(scores_for_this_word))]
            
            # log the POS we chose for word[j]
            pos_tags[j] = pos_for_this_word
        
        # end: for j in words_range
        
        # associate POS tags with words
        tagged_sent = [(words[j], pos_tags[j]) for j in words_range]
        
        return tagged_sent
        
    
    
    ######### `PRIVATE' FUNCTIONS #########
        
    def _cp_of_word_given_pos(self, word, pos):
        """
        Determine the conditional probability of a word given a POS
        
        :param word: string
        :param pos: string
        """
        
        # use the Conditional Frequency Distribution created by the trainer
        return self.words_given_pos[pos].freq(word)
        
    def _cp_of_pos2_given_pos1(self, pos2, pos1):
        """
        Determine the conditional probability of one POS given another.
        This is, in other words, the probability that POS2 follows POS1.
        
        :param pos2: string
        :param pos1: string
        """
        
        # use the Conditional Frequency Distribution created by the trainer
        return self.pos2_given_pos1[pos1].freq(pos2)
        
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
        
        # if we have all zeroes
        if max(value_array) == 0:
            # if we want to prefer one row, give it a value of 1
            if guess_index > -1:
                matrix[guess_index][j_value] = 1
                
            # otherwise, split the value of 1 over all rows
            else:
                for i in row_range:
                    matrix[i][j_value] = 1 / len(matrix)

        return matrix