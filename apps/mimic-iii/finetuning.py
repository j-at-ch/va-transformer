import os
import numpy as np

import methods
from data_utils import *
from arguments import Arguments
from models import FinetuningWrapper

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper


args = Arguments().parse(verbose=True)

# device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths

d_items_path = os.path.join(args.mimic_root, "d_items.csv")
train_path = os.path.join(args.data_root, "train_charts.pkl")
val_path = os.path.join(args.data_root, "val_charts.pkl")
mapping_path = os.path.join(args.data_root, "mappings.pkl")
ckpt_path = os.path.join(args.save_root, args.model_name)
logs_path = os.path.join(args.logs_root, "logs", args.model_name)

train_lbl_path = os.path.join(args.data_root, "train_labels.pkl")
val_lbl_path = os.path.join(args.data_root, "val_labels.pkl")

# fetch mappings

mappings_dict = fetch_mappings(mapping_path)
mappings = Mappings(mappings_dict)

# fetch labels  # NOTE: depends on specific label file format and names.

with open(train_lbl_path, 'rb') as f:
    X = pickle.load(f)
    train_labels_30 = {k: v['readm_30'] for k, v in  X['train_labels'].items()}
    train_labels_7 = {k: v['readm_7'] for k, v in  X['train_labels'].items()}
    del X

with open(val_lbl_path, 'rb') as f:
    X = pickle.load(f)
    val_labels_30 = {k: v['readm_30'] for k, v in  X['val_labels'].items()}
    val_labels_7 = {k: v['readm_7'] for k, v in  X['val_labels'].items()}
    del X

# generate datasets and loaders

data_train = fetch_data_as_torch(train_path, 'train_tokens')
data_val = fetch_data_as_torch(val_path, 'val_tokens')

ft_train_dataset = ClsSamplerDataset(data_train, args.seq_len, labels=train_labels_30)
ft_val_dataset = ClsSamplerDataset(data_val, args.seq_len, labels=val_labels_30)

ft_train_loader = cycle(DataLoader(ft_train_dataset, batch_size=args.ft_batch_size))
ft_val_loader = cycle(DataLoader(ft_val_dataset, batch_size=args.ft_batch_size))

# propensities


def propensity(di):
    x = sum(di.values()) / len(di)
    return x


p = propensity(train_labels_30)
weights = torch.tensor([p, 1 - p]).to(device)

# fetch model params

params_path = os.path.join(args.data_root, 'models', 'pre_model_exp1.pt')
X = torch.load(params_path, map_location=device)
states = X['model_state_dict']
base_states = { k[len('net.'):] if k[:len('net.')] == 'net.' else k : v for k, v in states.items()}

# initialisation of model

model = TransformerWrapper(
    num_tokens=mappings.num_tokens,
    max_seq_len=args.seq_len,
    attn_layers=Decoder(
        dim=args.attn_dim,
        depth=args.attn_depth,
        heads=args.attn_heads)
)

fit_model = FinetuningWrapper(model, num_classes=2,
                              state_dict=base_states,
                              weight=weights)
fit_model.to(device)
