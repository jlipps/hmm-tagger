######## PennTags.py ########

class PennTags:
    "A class to hold information about tags in the Penn Treebank"
    
    ######### CLASS VARIABLES #########

    proper_noun = 'NNP'
    common_noun = 'NN'
    pl_proper_noun = 'NNPS'
    pl_common_noun = 'NNS'
    verb_pp = 'VBN'
    verb = 'VB'
    det = 'DT'
    cardinal = 'CD'
    adv = 'RB'
    gerund = 'VBG'
    adj = 'JJ'
    prep = 'IN'
    comp_adv = 'RBR'
    comp_adj = 'JJR'
    verb_3s = 'VBZ'
    unknown = 'UNK'
    default = 'NN'
    default_upper = 'NNP'
    rare_tags = ['--','UNK'] # tags to add to POS list before testing