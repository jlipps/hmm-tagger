import nltk
import re

class TreebankCleaner:
    "A class for cleaning treebank text and turning it into an NLTK corpusReader object"
    
    def __init__(self, corpus_path, corpus_files):
        self.corpus_path = corpus_path
        self.corpus_files = corpus_files
        
    def clean(self):
        new_files = []
        for corpus_file in self.corpus_files:
            f = open(self.corpus_path + corpus_file, 'r')
            data = f.read()
            f.close()
            
            # remove trailing whitespaces
            data = re.sub(r' +(\r)?\n', '\n', data)
            #data = re.sub(r'^ +', '^', data)
            para_sep = r'======================================'
            data = re.sub(r'([^\.])(\n+)', '\\1 ', data)
            data = re.sub(para_sep, '\n'+para_sep+'\n', data)
            data = re.sub(r' +\n', '\n', data)
            data = re.sub(r'\n\n+', '\n', data)
            data = re.sub(para_sep + r'\n' + para_sep, para_sep, data)
            data = re.sub('^\n' + para_sep + '\n', '', data) # remove first para sep
            data = re.sub(r' *(\[|\]) *', ' ', data)
            data = re.sub(r'\n +', '\n', data)
            data = re.sub(r'^ +', '', data)
            #optional para sep removal
            data = re.sub(para_sep + r'\n', '', data)
            
            new_file = corpus_file + '_cleaned'
            f = open(self.corpus_path + new_file, 'w')
            f.write(data)
            