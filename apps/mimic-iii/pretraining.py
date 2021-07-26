import os
import sys
from pprint import pprint

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# repo imports

import methods
from data_utils import *
from arguments import Arguments


def pretrain(args):
    print('*'*17, 'chart-transformer called for training with the following settings:', sep='\n')
    pprint(vars(args), indent=2)

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

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings_dict, d_items_df)

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

    # wrap for autoregressive

    pre_model = AutoregressiveWrapper(model)
    pre_model.to(device)

    # load data for pretraining based on arguments

    train_dataset = ClsSamplerDataset(data_train, args.seq_len, device)
    val_dataset = ClsSamplerDataset(data_val, args.seq_len, device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    # code for testing if needed
    #train_loader = [X for i, X in enumerate(train_loader) if i < 2]
    #val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    train_cycler = cycle(train_loader)
    val_cycler = cycle(val_loader)

    optim = torch.optim.Adam(pre_model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = methods.TrainingMethods(pre_model, writer)

    # training loop

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        ________ = training.train(train_loader, optim, epoch, batch_size=args.batch_size_tr)
        val_loss = training.evaluate(val_loader, epoch, batch_size=args.batch_size_val)

        if val_loss < best_val_loss:
            print("Saving checkpoint...")
            torch.save({
                'train_epoch': epoch,
                'model_state_dict': pre_model.state_dict(),
                'args': vars(args),
                'seq_len': args.seq_len,
                'optim_state_dict': optim.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)
            print("Checkpoint saved!\n")
            best_val_loss = val_loss

        pre_model.eval()
        with torch.no_grad():
            tokens = torch.tensor(np.arange(0, 200), dtype=torch.int)
            X = torch.zeros(200, dtype=torch.int)
            X[0:len(tokens)] = tokens
            X = X.to(device)
            Z = pre_model.net.token_emb(X)
            metadata = [label for label in map(labeller.token2label, X.numpy())]
            writer.add_embedding(Z,
                                 metadata=metadata,
                                 global_step=epoch,
                                 tag='token embeddings')

        print(f'epoch {epoch} completed!')
        print('flushing writer...')
        writer.flush()

    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments(mode='pretraining').parse()
    pretrain(arguments)
