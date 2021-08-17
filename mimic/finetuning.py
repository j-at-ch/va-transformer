import sys
from pprint import pprint
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from z_transformers.transformers import TransformerWrapper, Decoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import methods
from data_utils import *
from arguments import Arguments
from models import FinetuningWrapper


def finetune(args):
    print('*' * 17, 'chart-transformer summoned for finetuning with the following settings:', sep='\n')
    pprint(vars(args), indent=2)

    # paths

    d_items_path = os.path.join(args.data_root, "D_LABITEMS.csv")
    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)

    train_lbl_path = os.path.join(args.data_root, "train_targets.pkl")
    val_lbl_path = os.path.join(args.data_root, "val_targets.pkl")
    params_path = os.path.join(args.model_root, args.pretuned_model)

    # device

    device = torch.device(args.device)

    # fetch mappings

    mappings_dict = fetch_mappings(mapping_path)
    mappings = Mappings(mappings_dict)

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings_dict, d_items_df)

    # fetch labels

    with open(train_lbl_path, 'rb') as f:
        X = pickle.load(f)
        train_targets = {k: v[args.label_set] for k, v in X['train_targets'].items()}
        del X

    with open(val_lbl_path, 'rb') as f:
        X = pickle.load(f)
        val_targets = {k: v[args.label_set] for k, v in X['val_targets'].items()}
        del X

    # generate datasets and loaders

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    ft_train_dataset = ClsSamplerDataset(data_train, args.seq_len, device, labels=train_targets)
    ft_val_dataset = ClsSamplerDataset(data_val, args.seq_len, device, labels=val_targets)

    ft_train_loader = DataLoader(ft_train_dataset, batch_size=args.ft_batch_size, shuffle=True)
    ft_val_loader = DataLoader(ft_val_dataset, batch_size=args.ft_batch_size, shuffle=True)

    ft_train_cycler = cycle(ft_train_loader)
    ft_val_cycler = cycle(ft_val_loader)

    #  for quick test run

    if args.test_run:
        ft_train_loader = [X for i, X in enumerate(ft_train_loader) if i < 2]
        ft_val_loader = [X for i, X in enumerate(ft_val_loader) if i < 2]

    # propensities

    def propensity(di):
        return sum(di.values()) / len(di)

    p = propensity(train_targets)
    print(f"Train set positive class propensity is {p}")

    if args.weighted_loss:
        weights = torch.tensor([p, 1 - p]).to(device)
    else:
        weights = None

    # fetch model params

    X = torch.load(params_path, map_location=device)
    states = X['model_state_dict']

    # base_states = {k[len('net.'):] if k[:len('net.')] == 'net.' else k: v for k, v in states.items()}

    # initialisation of model

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        max_seq_len=args.seq_len,  # NOTE: max_seq_len necessary for the absolute positional embeddings.
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout)
    )

    fit_model = FinetuningWrapper(model, num_classes=2,
                                  seq_len=args.seq_len,
                                  state_dict=states,
                                  load_from_pretuning=True,
                                  weight=weights)
    fit_model.to(device)

    # initialise optimiser

    optimizer = torch.optim.Adam(fit_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = methods.FinetuningMethods(fit_model, writer)

    # write initial embeddings

    if args.write_initial_embeddings:
        print("Writing initial token embeddings to writer...")
        training.write_embeddings(0, mappings, labeller, args.seq_len, device)
        print("Initial token embeddings written!")

    # training loop

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        ________ = training.train(ft_train_loader, optimizer, epoch)
        val_loss = training.evaluate(ft_val_loader, epoch)

        # whether to checkpoint model

        if val_loss < best_val_loss:
            print("Saving checkpoint...")
            torch.save({
                'train_epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
                'model_state_dict': fit_model.state_dict(),
                'optim_state_dict': optimizer.state_dict()
            }, ckpt_path)

            # track checkpoint's embeddings
            if args.write_embeddings:
                print("Writing checkpoint's token embeddings to writer...")
                training.write_embeddings(epoch + 1, mappings, labeller, args.seq_len, device)
                print("Checkpoint's token embeddings written!")

            print("Checkpoint saved!\n")
            best_val_loss = val_loss

        # update scheduler

        scheduler.step()

        # tracking model classification metrics for val set

        ________ = training.predict(ft_train_loader, epoch, device, prefix="train")
        ________ = training.predict(ft_val_loader, epoch, device, prefix="val")

        # flushing writer

        print(f'epoch {epoch} completed!')
        print('flushing writer...')
        writer.flush()
    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments(mode='finetuning').parse()

    # check output roots exist; if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # run finetuning

    finetune(arguments)
