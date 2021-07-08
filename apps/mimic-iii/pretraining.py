import os
import pickle
import random
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# paths

d_items_path = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4/d_items.csv"
data_root = "C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers"
train_path = os.path.join(data_root, "train_charts.pkl")
val_path = os.path.join(data_root, "val_charts.pkl")
mapping_path = os.path.join(data_root, "mappings.pkl")
ckpt_path = os.path.join(data_root, "model.pt")

# misc

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

def fetch_data(path, var_key):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data[var_key]


trX = fetch_data(train_path, 'train_tokens')
vaX = fetch_data(val_path, 'val_tokens')

data_train = {k: torch.from_numpy(v) for k, v in trX.items()}
data_val = {k: torch.from_numpy(v) for k, v in vaX.items()}


# yield from loader

def cycle(loader):
    while True:
        for data in loader:
            yield data


# constants  # TODO: consider having these stored in checkpoint

NUM_BATCHES = 1000 #int(1e5)
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

class SeqSamplerDataset(Dataset):  # TODO: tidy __getitem__ method
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

# training loop








for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    # validate model

    best_val_loss = np.inf
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_loss = model(next(val_loader))
            print(f'validation loss: {val_loss.item()}')

        # checkpoint model

        if (i > CHECKPOINT_AFTER) & (val_loss.item() < best_val_loss):
            torch.save({
                'train_step': i,
                'model_state_dict': model.state_dict(),
                'SEQ_LEN': SEQ_LEN,
                'optim_state_dict': optim.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)
            print("Checkpoint saved!\n")

    # generate sequence

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        primer_str = decode_tokens(inp.numpy())
        print('primer:', primer_str, '*' * 100, sep='\n')

        sample = model.generate(inp, GENERATE_LENGTH)
        sample_str = decode_tokens(sample.numpy())
        print('output:', sample_str, sep='\n')
