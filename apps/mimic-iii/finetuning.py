import os
import numpy as np
from pprint import pprint

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from x_transformers import TransformerWrapper, Decoder

import methods
from data_utils import *
from arguments import Arguments
from models import FinetuningWrapper


def finetune(args)
    print('*' * 17, 'chart-transformer called for finetuning with the following settings:', sep='\n')
    pprint(vars(args), indent=2)

    # paths

    d_items_path = os.path.join(args.mimic_root, "d_items.csv")
    train_path = os.path.join(args.data_root, "train_charts.pkl")
    val_path = os.path.join(args.data_root, "val_charts.pkl")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")
    ckpt_path = os.path.join(args.save_root, "models", args.model_name)
    logs_path = os.path.join(args.logs_root, "logs", args.model_name)

    train_lbl_path = os.path.join(args.data_root, "train_labels.pkl")
    val_lbl_path = os.path.join(args.data_root, "val_labels.pkl")
    params_path = os.path.join(args.data_root, 'models', args.pretuned_model)

    #device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fetch mappings

    mappings_dict = fetch_mappings(mapping_path)
    mappings = Mappings(mappings_dict)

    # fetch labels

    with open(train_lbl_path, 'rb') as f:
        X = pickle.load(f)
        train_labels = {k: v[args.label_set] for k, v in X['train_labels'].items()}
        del X

    with open(val_lbl_path, 'rb') as f:
        X = pickle.load(f)
        val_labels = {k: v[args.label_set] for k, v in X['val_labels'].items()}
        del X

    # generate datasets and loaders

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    ft_train_dataset = ClsSamplerDataset(data_train, args.seq_len, device, labels=train_labels)
    ft_val_dataset = ClsSamplerDataset(data_val, args.seq_len, device, labels=val_labels)

    ft_train_loader = cycle(DataLoader(ft_train_dataset, batch_size=args.ft_batch_size, shuffle=True))
    ft_val_loader = cycle(DataLoader(ft_val_dataset, batch_size=args.ft_batch_size, shuffle=True))

    # propensities


    def propensity(di):
        x = sum(di.values()) / len(di)
        return x


    p = propensity(train_labels)
    weights = torch.tensor([p, 1 - p]).to(device)

    # fetch model params

    X = torch.load(params_path, map_location=device)
    states = X['model_state_dict']
    base_states = {k[len('net.'):] if k[:len('net.')] == 'net.' else k: v for k, v in states.items()}

    # initialisation of model

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        max_seq_len=args.seq_len,  # NOTE: max_seq_len necessary for the absolute positional embeddings.
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads)
    )

    fit_model = FinetuningWrapper(model, num_classes=2,
                                  seq_len=args.seq_len,
                                  state_dict=base_states,
                                  weight=weights)
    fit_model.to(device)

    # initialise optimiser

    optim = torch.optim.Adam(fit_model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = methods.FinetuningMethods(fit_model, writer)

    # training loop

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        ________ = training.train(ft_train_loader, optim, epoch,
                                  num_batches=args.num_batches_tr, batch_size=args.batch_size_tr)
        val_loss = training.evaluate(ft_val_loader, epoch,
                                     num_batches=args.num_batches_val, batch_size=args.batch_size_val)

        if val_loss < best_val_loss:
            print("Saving checkpoint...")
            torch.save({
                'train_epoch': epoch,
                'model_state_dict': fit_model.state_dict(),
                'args': vars(args),
                'seq_len': args.seq_len,
                'optim_state_dict': optim.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)
            print("Checkpoint saved!\n")
            best_val_loss = val_loss

        print(f'epoch {epoch} completed!')
        print('flushing writer...')
        writer.flush()

    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments().parse()
    finetune(arguments)