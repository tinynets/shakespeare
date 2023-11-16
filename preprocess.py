"""
This file is used to preprocess the data and clean it in preparation for training in a neural network
"""
import os
from torchtext.data.utils import get_tokenizer

class Preprocess:
    def __init__(self, datadir):
        self.data_dir = datadir
        self.data = self.load_data()
        self.preprocess_data()
        self.tokens = []

    def load_data(self):
        fnames = os.listdir(self.data_dir)
        lines = []
        for f in fnames:
            with open(os.path.join(self.data_dir, f)) as f:
                file_lines = f.readlines()
                lines.extend(file_lines)
        return lines
    
    def analyze_data(self):


        return
    
    def preprocess_data(self):
        lines = [line.strip().lower() for line in self.data if line.strip()]
        self.data = lines


        