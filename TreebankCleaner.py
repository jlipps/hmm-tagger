######### TreebankCleaner.py #########

import re # for regular expressions
from Helper import msg # for messaging

class TreebankCleaner:
    "A class for cleaning treebank text"
    
    def __init__(self, corpus_path, corpus_files):
        """
        Initialize a TreebankCleaner object.
        
        :param corpus_path: path of corpus files
        :param corpus_files: list of corpus files
        """
        
        self.corpus_path = corpus_path
        self.corpus_files = corpus_files
    
    ######### `PUBLIC' FUNCTIONS #########
        
    def clean(self):
        """
        Clean corpus files and write the results to disk
        """
        
        # loop through files
        for corpus_file in self.corpus_files:
            
            msg("Cleaning %s..." % corpus_file)
            
            # get the file in a string
            f = open(self.corpus_path + corpus_file, 'r')
            data = f.read()
            f.close()
            
            # use an unoptimized set of arcane regular expressions to clean the data
            data = re.sub(r' +(\r)?\n', '\n', data)
            para_sep = r'======================================'
            data = re.sub(r'([^\.])(\n+)', '\\1 ', data)
            data = re.sub(para_sep, '\n'+para_sep+'\n', data)
            data = re.sub(r' +\n', '\n', data)
            data = re.sub(r'\n\n+', '\n', data)
            data = re.sub(para_sep + r'\n' + para_sep, para_sep, data)
            data = re.sub('^\n' + para_sep + '\n', '', data)
            data = re.sub(r' *(\[|\]) *', ' ', data)
            data = re.sub(r'\n +', '\n', data)
            data = re.sub(r'^ +', '', data)
            data = re.sub(para_sep + r'\n', '', data)
            
            # write the cleaned data to a new file
            new_file = corpus_file + '_cleaned'
            f = open(self.corpus_path + new_file, 'w')
            f.write(data)
            
            msg("done!\n")
