"""
This file is used to preprocess the data and clean it in preparation for training in a neural network
"""
import os
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

class Preprocess:
    def __init__(self, datadir):
        self.data_dir = datadir
        self.raw = None
        self.data = self.load_data()
        self.tokens = []
        self.preprocess_data()
        self.vocab = build_vocab_from_iterator([self.tokens])
        self.analysis = self.analyze_data()
        self.numerical_tokens = [self.vocab[token] for token in self.tokens]
        

    def load_data(self):
        fnames = os.listdir(self.data_dir)
        lines = []
        raw_lines = []
        for f in fnames:
            with open(os.path.join(self.data_dir, f)) as file:
                file_lines = file.readlines()
                raw_lines.extend(file_lines)
                lines.extend(file_lines)
        self.raw = raw_lines
        return lines
    
    def analyze_data(self):
        analysis = {}
        
        clean_data = [line.strip().lower() for line in self.data if line.strip() != '']
        line_lengths = [len(line) for line in clean_data]
        analysis["avg_len"] = np.mean(line_lengths)
        analysis["word_counts"] = Counter(self.tokens)

        return analysis
    
    def preprocess_data(self):
        tokenizer = get_tokenizer('basic_english')
        lines = [line.strip().lower() for line in self.data if line.strip()]
        tokens = [token for line in lines for token in tokenizer(line)]
        self.tokens = tokens

    # defining here because one may want to run analysis and decide after viewing the data
    def set_padding_size(self, size):
        self.padding_size = size


        