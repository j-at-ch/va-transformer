import os
import pickle
import random
import tqdm
import numpy as np
import torch

import methods
import data_utils
from arguments import Arguments

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper


# parse arguments

args = Arguments().parse(verbose=True)

# paths

#mimic_root = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4"
#data_root = "C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers"
#save_root = "C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers"

d_items_path = os.path.join(args.mimic_root, "d_items.csv")
train_path = os.path.join(args.data_root, "train_charts.pkl")
val_path = os.path.join(args.data_root, "val_charts.pkl")
mapping_path = os.path.join(args.data_root, "mappings.pkl")
ckpt_path = os.path.join(args.save_root, "model.pt")
logs_path = os.path.join(args.save_root, "tensorboard_logs", "logs")

# device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# token mappings:  # TODO: refactor to module where possible.

with open(mapping_path, 'rb') as f:
    mappings = pickle.load(f)
    itemid2token = mappings['itemid2token']
    token2itemid = mappings['token2itemid']
    del mappings

num_tokens = len(itemid2token)

# token mappings: decoders


def decode_token(token):
    return str(token2itemid[token])


def decode_tokens(tokens):
    return ' '.join(list(map(decode_token, tokens)))


# get data

def fetch_data_as_torch(path, var_key):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    di = data[var_key]
    return {k: torch.from_numpy(v) for k, v in di.items()}


data_train = fetch_data_as_torch(train_path, 'train_tokens')
data_val = fetch_data_as_torch(val_path, 'val_tokens')

# yield from loader

def cycle(loader):
    while True:
        for data in loader:
            yield data


# constants  # TODO: consider having these stored in checkpoint

NUM_EPOCHS = 2
NUM_BATCHES = 100
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4  # 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 10
CHECKPOINT_AFTER = 10
GENERATE_EVERY = 20
GENERATE_LENGTH = 200
SEQ_LEN = 200

# instantiate GPT-like decoder model

model = TransformerWrapper(
    num_tokens=num_tokens,  # 256. Note - expects each val in data to be [0, num_tokens)
    max_seq_len=SEQ_LEN,
    attn_layers=Decoder(dim=100, depth=3, heads=4)  # 512, 6, 8
)

model = AutoregressiveWrapper(model)
model.to(device)


# custom sequence-excerpt sampler

class SeqSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()  # gives access to Dataset methods.
        self.data = data
        self.seq_len = seq_len
        self.lookup = dict(zip(np.arange(len(self.data)),
                               self.data.keys()))

    def __getitem__(self, key):  # a.t.m. when data[key] shorter length than SEQ_LEN, padded with 0.
        item_len = self.data[self.lookup[key]].size(0)
        rand_start = torch.randint(0, item_len - self.seq_len, (1,)) if item_len > self.seq_len else 0
        lenfromseq = min(item_len, self.seq_len)
        sample = torch.zeros(self.seq_len)
        sample[:lenfromseq] = self.data[self.lookup[key]][rand_start: rand_start + lenfromseq]
        sample = sample.long()
        return sample.to(device)

    def __len__(self):
        return len(self.data)


train_dataset = SeqSamplerDataset(data_train, SEQ_LEN)
val_dataset   = SeqSamplerDataset(data_val, SEQ_LEN)

train_loader  = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset,   batch_size=BATCH_SIZE))

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter(log_dir=logs_path)
training = methods.TrainingMethods(model, writer)

# training loop

for epoch in range(1, NUM_EPOCHS + 1):
    training.train(train_loader, optim, epoch, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE)
    training.evaluate(val_loader, epoch, num_batches=100, batch_size=4)

writer.close()
