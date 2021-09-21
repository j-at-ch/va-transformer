import torch
import numpy as np
from torch.utils.data import Dataset


class VgSamplerDataset(Dataset):
    def __init__(self,
                 tokens,
                 seq_len,
                 mappings,
                 device,
                 quantiles=None,
                 targets=None,
                 specials=None,
                 align_sample_at='random/SOS'
                 ):
        super().__init__()
        self.tokens = tokens
        self.seq_len = seq_len
        self.mappings = mappings
        self.device = device
        self.quantiles = quantiles
        self.targets = targets
        self.specials = specials
        self.align_sample_at = align_sample_at
        self.lookup = dict(zip(np.arange(len(self.tokens)), self.tokens.keys()))

    @staticmethod
    def add_specials_(seq, specials, sos_token, eos_token):
        if specials == 'EOS':
            new_seq = torch.cat((seq, torch.tensor([eos_token])), 0)
        elif specials == 'both':
            new_seq = torch.cat((torch.tensor([sos_token]), seq,
                                 torch.tensor([eos_token])), 0)
        elif specials == 'SOS':
            new_seq = torch.cat((torch.tensor([sos_token]), seq), 0)
        else:
            raise Exception('Unknown specials configuration specified!')
        return new_seq

    def __getitem__(self, key):
        index = self.lookup[key]
        token_seq = self.tokens[index]

        if self.specials is not None:
            token_seq = self.add_specials_(self.tokens[index],
                                           self.specials,
                                           self.mappings.sos_token,
                                           self.mappings.eos_token)

        item_len = token_seq.size(0)
        obtainable_len = min(item_len, self.seq_len)

        # extract sample

        sample = self.mappings.pad_token * torch.ones(self.seq_len)
        if self.align_sample_at == 'SOS':
            start_index = 0
            end_index = start_index + obtainable_len
            sample[:obtainable_len] = token_seq[start_index: end_index]
        elif self.align_sample_at == 'EOS':
            end_index = item_len
            start_index = max(0, end_index - self.seq_len)
            sample[self.seq_len - obtainable_len:self.seq_len] = token_seq[start_index: end_index]
        elif self.align_sample_at == 'random/SOS':
            start_index = torch.randint(0, item_len - self.seq_len, (1,)) if item_len > self.seq_len else 0
            end_index = start_index + obtainable_len
            sample[:obtainable_len] = token_seq[start_index: end_index]
        elif self.align_sample_at == 'random/EOS':
            end_index = torch.randint(self.seq_len, item_len, (1,)) if item_len > self.seq_len else item_len
            start_index = max(0, end_index - self.seq_len)
            sample[self.seq_len - obtainable_len:self.seq_len] = token_seq[start_index: end_index]
        sample = sample.long().to(self.device)

        # extract guides and targets if required

        if self.quantiles is not None:
            if self.specials:
                guide_seq = self.add_specials_(self.quantiles[index],
                                               self.specials,
                                               self.mappings.sos_guide_token,
                                               self.mappings.eos_guide_token)
            else:
                guide_seq = self.quantiles[index]

            guide_sample = self.mappings.pad_guide_token * torch.ones(self.seq_len)

            if self.align_sample_at in ['EOS', 'random/EOS']:
                guide_sample[self.seq_len - obtainable_len: self.seq_len] = guide_seq[start_index: end_index]
            else:
                guide_sample[:obtainable_len] = guide_seq[start_index: end_index]
            guide_sample = guide_sample.long().to(self.device)

        if self.targets is not None:
            targets = torch.tensor(self.targets[index])
            targets = targets.long().to(self.device)

        if (self.quantiles is None) & (self.targets is None):
            return sample
        elif (self.quantiles is not None) & (self.targets is None):
            return sample, guide_sample
        elif (self.quantiles is None) & (self.targets is not None):
            return sample, targets
        else:
            return sample, guide_sample, targets

    def __len__(self):
        return len(self.tokens)


def cycler(loader):
    while True:
        for data in loader:
            yield data
