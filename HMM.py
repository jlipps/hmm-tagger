class HMM:
    "A class for building Hidden Markov Models of tagged word data"
    
    def __init__(self, untagged_sents, pos_tags, words_given_pos, pos2_given_pos1):
        self.default_tag = 'UNK'
        self.start_tag = '^'
        self.untagged_sents = untagged_sents
        self.pos_tags = pos_tags
        self.words_given_pos = words_given_pos
        self.pos2_given_pos1 = pos2_given_pos1
        
    def tag(self):
        sents = self.tag_sent(self.untagged_sents[0])
        print sents
        return sents
        #return [self.tag_sent(sent) for sent in self.untagged_sents]
        
    def tag_sent(self, untagged_sent):
        num_words = len(untagged_sent)
        num_pos = len(self.pos_tags)
        score = [[None for j in range(num_words)] for i in range(num_pos)]
        backpointer = [[None for j in range(num_words)] for i in range(num_pos)]
        pos_tags = ['' for i in range(num_words)]
        cpwp = self._cp_of_word_given_pos
        cpp2p1 = self._cp_of_pos2_given_pos1
        # initialize
        for i in range(num_pos):
            score[i][0] = cpwp(untagged_sent[0], self.pos_tags[i]) * cpp2p1(self.pos_tags[i], '^')
            backpointer[i][0] = 0
            print score[i]
        
        for j in range(1,num_words):
            for i in range(num_pos):
                tmp_scores = [score[k][j-1] * cpp2p1(self.pos_tags[i], self.pos_tags[k]) * cpwp(untagged_sent[j], self.pos_tags[i]) for k in range(num_pos-1)]
                score[i][j] = max(tmp_scores)
                backpointer[i][j] = tmp_scores.index(score[i][j])
        
        pos_tags[num_words-1] = max([score[i][num_words-1] for i in range(num_pos)])
        for i in range(num_words-2, 0):
            pos_tags[i] = backpointer[pos_tags[i+1]][i+1]
        tagged_sent = []
        
        return [(untagged_sent[i], pos_tags[i]) for i in range(num_words)]
    
    
    # `PRIVATE' FUNCTIONS
        
    def _cp_of_word_given_pos(self, word, pos):
        return self.words_given_pos[pos].freq(word)
        
    def _cp_of_pos2_given_pos1(self, pos2, pos1):
        return self.pos2_given_pos1[pos1].freq(pos2)