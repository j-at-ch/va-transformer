import numpy as np
import torch
from torch.utils.data import Dataset


class BasicSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device, labels=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.device = device
        self.seq_len = seq_len
        self.lookup = dict(zip(np.arange(len(self.data)), self.data.keys()))

    def __getitem__(self, key):
        index = self.lookup[key]
        item_len = self.data[index].size(0)
        rand_start = torch.randint(0, item_len - self.seq_len, (1,)) if item_len > self.seq_len else 0
        lenfromseq = min(item_len, self.seq_len)
        sample = torch.zeros(self.seq_len)
        sample[:lenfromseq] = self.data[index][rand_start: rand_start + lenfromseq]

        if self.labels is not None:
            label = torch.tensor(self.labels[index])
            return sample.long().to(self.device), label.long().to(self.device)
        else:
            return sample.long().to(self.device)

    def __len__(self):
        return len(self.data)


class VgSamplerDataset(Dataset):
    def __init__(self,
                 tokens,
                 seq_len,
                 mappings,
                 device,
                 quantiles=None,
                 labels=None,
                 use_specials=False,
                 align_sample_at='random'
                 ):
        super().__init__()
        self.tokens = tokens
        self.seq_len = seq_len
        self.mappings = mappings
        self.device = device
        self.quantiles = quantiles
        self.labels = labels
        self.use_specials = use_specials
        self.align_sample_at = align_sample_at
        self.lookup = dict(zip(np.arange(len(self.tokens)), self.tokens.keys()))

    @staticmethod
    def add_specials_(seq, sos_token, eos_token):
        return torch.cat((torch.tensor([sos_token]), seq, torch.tensor([eos_token])), 0)

    def __getitem__(self, key):
        index = self.lookup[key]
        if self.use_specials:
            self.tokens[index] = self.add_specials_(self.tokens[index],
                                                    self.mappings.sos_token,
                                                    self.mappings.eos_token)
            if self.quantiles is not None:
                self.quantiles[index] = self.add_specials_(self.quantiles[index],
                                                           self.mappings.sos_guide_token,
                                                           self.mappings.eos_guide_token)
        item_len = self.tokens[index].size(0)
        obtainable_len = min(item_len, self.seq_len)

        # extract sample

        sample = self.mappings.pad_token * torch.ones(self.seq_len)
        if self.align_sample_at == 'SOS':
            start_index = 0
            end_index = start_index + obtainable_len
            sample[:obtainable_len] = self.tokens[index][start_index: end_index]
        elif self.align_sample_at == 'EOS':
            end_index = item_len
            start_index = max(0, end_index - self.seq_len)
            sample[self.seq_len - obtainable_len:self.seq_len] = self.tokens[index][start_index: end_index]
        else:
            start_index = torch.randint(0, item_len - self.seq_len, (1,)) if item_len > self.seq_len else 0
            end_index = start_index + obtainable_len
            sample[:obtainable_len] = self.tokens[index][start_index: end_index]
        sample = sample.long().to(self.device)

        # extract guides and labels if required

        if self.quantiles is not None:
            quantiles = self.mappings.pad_guide_token * torch.ones(self.seq_len)
            if self.align_sample_at == 'EOS':
                quantiles[self.seq_len - obtainable_len: self.seq_len] = self.quantiles[index][start_index: end_index]
            else:
                quantiles[:obtainable_len] = self.quantiles[index][start_index: end_index]
            quantiles = quantiles.long().to(self.device)

        if self.labels is not None:
            labels = torch.tensor(self.labels[index])
            labels = labels.long().to(self.device)

        if (self.quantiles is None) & (self.labels is None):
            return sample
        elif (self.quantiles is not None) & (self.labels is None):
            return sample, quantiles
        elif (self.quantiles is None) & (self.labels is not None):
            return sample, labels
        else:
            return sample, quantiles, labels

    def __len__(self):
        return len(self.tokens)


def cycle(loader):
    while True:
        for data in loader:
            yield data