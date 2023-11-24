"""
This file will prepare the dataset for training and get it ready to be passed into the DataLoader
"""

import torch
from torch.utils.data import Dataset

class ShakesPeareDataset(Dataset):
    def __init__(self, data, sequence_length=5, vocab=None):
        self.padding_token  = "<pad>"
        self.data = data
        self.sequence_length = sequence_length
        self.vocab = vocab

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.sequence_length]
        padding_size = max(0, self.sequence_length - len(seq))
        seq += [self.padding_token] * padding_size

        X_tokens = seq[:-1]
        y_tokens = seq[1:]

        X = torch.tensor([self.vocab[token] for token in X_tokens])
        y = torch.tensor([self.vocab[token] for token in y_tokens])
        
        return X, y

