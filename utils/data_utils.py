import numpy as np
import pickle
import torch
import os

from torch.utils.data import Dataset


class ClsSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device, labels=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.device = device
        self.seq_len = seq_len
        self.lookup = dict(zip(np.arange(len(self.data)), self.data.keys()))
        self.counter = 0

    def __getitem__(self, key):
        self.counter += 1

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
                 device,
                 quantiles=None,
                 labels=None,
                 token_pad_value=0,
                 quantile_pad_value=-1):
        super().__init__()
        self.tokens = tokens
        self.quantiles = quantiles
        self.labels = labels
        self.device = device
        self.seq_len = seq_len
        self.lookup = dict(zip(np.arange(len(self.tokens)), self.tokens.keys()))
        self.token_pad_value = token_pad_value
        self.quantile_pad_value = quantile_pad_value

    def __getitem__(self, key):
        index = self.lookup[key]
        item_len = self.tokens[index].size(0)
        rand_start = torch.randint(0, item_len - self.seq_len, (1,)) if item_len > self.seq_len else 0
        len_from_seq = min(item_len, self.seq_len)
        sample = self.token_pad_value * torch.ones(self.seq_len)
        sample[:len_from_seq] = self.tokens[index][rand_start: rand_start + len_from_seq]
        sample = sample.long().to(self.device)

        if self.quantiles is not None:
            quantiles = self.quantile_pad_value * torch.ones(self.seq_len)
            quantiles[:len_from_seq] = self.quantiles[index][rand_start: rand_start + len_from_seq]
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


class Mappings:
    def __init__(self, mappings, pad_token=None, clf_token=None, eos_token=None):
        self.itemid2token = mappings['itemid2token']
        self.token2itemid = mappings['token2itemid']
        self.token2trcount = mappings['token2trcount']
        self.guide_name2guide_token = {
            'CAT': 0, 'XLOW': 1, 'LOW': 2, 'MID': 3, 'HIGH': 4, 'XHIGH': 5, 'PAD': 6
        }  # todo incorporate into preprocessing pipeline
        self.num_tokens = len(self.itemid2token)
        self.num_guide_tokens = len(self.guide_name2guide_token)
        self.pad_token = pad_token
        self.clf_token = clf_token
        self.eos_token = eos_token

        if pad_token is not None:
            self.append_special_('[PAD]', pad_token)
        if clf_token is not None:
            self.append_special_('[CLF]', clf_token)
        if eos_token is not None:
            self.append_special_('[EOS]', eos_token)

    def append_special_(self, name, token):
        self.itemid2token[name] = token
        self.token2itemid[token] = name

    def decode_token(self, token):
        return str(self.token2itemid[token])

    def decode_tokens(self, tokens):
        return ' '.join(list(map(self.decode_token, tokens)))

    def topNtokens_tr(self, N):
        d = sorted(self.token2trcount.items(), key=lambda item: item[1], reverse=True)
        return dict(d[0:N])


class Labellers(Mappings):
    def __init__(self, mappings, d_items_df):
        super().__init__(mappings)
        self.mappings = mappings
        self.d_items_df = d_items_df

    def token2label(self, token):
        if token == 0:
            return '[PAD]'
        else:
            itemid = self.token2itemid[token]
            x = self.d_items_df.loc[itemid, 'LABEL']
        return x

    def tokens2labels(self, tokens):
        return '\n\t -> '.join(list(map(self.token2label, tokens)))


def fetch_mappings(mapping_path):
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    return mappings


def fetch_data_as_torch(path, var_key):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    di = data[var_key]
    return {k: torch.from_numpy(v) for k, v in di.items()}


def retrieve_model_args(data_root, model_name, device):
    params_path = os.path.join(data_root, 'models', model_name)
    model_dict = torch.load(params_path, map_location=device)
    return model_dict
