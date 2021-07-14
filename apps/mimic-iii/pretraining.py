import os
import numpy as np

import methods
from data_utils import *
from arguments import Arguments

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# parse arguments

args = Arguments().parse(verbose=True)

# paths

d_items_path = os.path.join(args.mimic_root, "d_items.csv")
train_path = os.path.join(args.data_root, "train_charts.pkl")
val_path = os.path.join(args.data_root, "val_charts.pkl")
mapping_path = os.path.join(args.data_root, "mappings.pkl")
ckpt_path = os.path.join(args.save_root, args.model_name)
logs_path = os.path.join(args.logs_root, "logs", args.model_name)

# device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mappings

mappings_dict = fetch_mappings(mapping_path)
mappings = Mappings(mappings_dict)

# get data

data_train = fetch_data_as_torch(train_path, 'train_tokens')
data_val = fetch_data_as_torch(val_path, 'val_tokens')

# instantiate GPT-like decoder architecture

model = TransformerWrapper(
    num_tokens=mappings.num_tokens,
    max_seq_len=args.seq_len,
    attn_layers=Decoder(
        dim=args.attn_dim,
        depth=args.attn_depth,
        heads=args.attn_heads)
)

pre_model = AutoregressiveWrapper(model)
pre_model.to(device)

train_dataset = ClsSamplerDataset(data_train, args.seq_len, device)
val_dataset = ClsSamplerDataset(data_val, args.seq_len, device)

train_loader = cycle(DataLoader(train_dataset, batch_size=args.batch_size_tr))
val_loader = cycle(DataLoader(val_dataset, batch_size=args.batch_size_val))

optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
training = methods.TrainingMethods(pre_model, writer)

# training loop
best_val_loss = np.inf
for epoch in range(args.num_epochs):
    training.train(train_loader, optim, epoch, num_batches=args.num_batches_tr, batch_size=args.batch_size_tr)
    val_loss = training.evaluate(val_loader, epoch, num_batches=args.num_batches_val, batch_size=args.batch_size_val)

    if val_loss < best_val_loss:
        print("Saving checkpoint...")
        torch.save({
            'train_epoch': epoch,
            'model_state_dict': pre_model.state_dict(),
            'args': vars(args),
            'SEQ_LEN': args.seq_len,
            'optim_state_dict': optim.state_dict(),
            'val_loss': val_loss
        }, ckpt_path)
        print("Checkpoint saved!\n")
        best_val_loss = val_loss
    print(f'epoch {epoch} completed!')
    print('flushing writer...')
    writer.flush()

writer.close()
