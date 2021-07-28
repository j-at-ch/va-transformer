import numpy as np
import pickle
import torch
import os

from torch.utils.data import DataLoader, Dataset


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


def cycle(loader):
    while True:
        for data in loader:
            yield data


class Mappings:
    def __init__(self, mappings):
        self.itemid2token = mappings['itemid2token']
        self.token2itemid = mappings['token2itemid']
        self.num_tokens = len(self.itemid2token)

    def decode_token(self, token):
        return str(self.token2itemid[token])

    def decode_tokens(self, tokens):
        return ' '.join(list(map(self.decode_token, tokens)))


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
