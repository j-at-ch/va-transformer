import sys

import numpy as np
import pandas as pd
from pprint import pprint
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import model_methods
from utils.data_utils import *
from utils.arguments import Arguments


def baseline(args):
    print('*' * 17, 'summoning baseline models for classification with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)
    mapping_path = os.path.join(args.data_root, "mappings.pkl")

    train_lbl_path = os.path.join(args.data_root, "train_targets.pkl")
    val_lbl_path = os.path.join(args.data_root, "val_targets.pkl")

    # device

    device = torch.device(args.device)

    # mappings

    mappings_dict = fetch_mappings(mapping_path)
    mappings = Mappings(mappings_dict)
    num_tokens = mappings.num_tokens
    num_quantiles = 7

    # fetch labels

    with open(train_lbl_path, 'rb') as f:
        X = pickle.load(f)
        train_targets = {k: v[args.label_set] for k, v in X['train_targets'].items()}
        del X

    with open(val_lbl_path, 'rb') as f:
        X = pickle.load(f)
        val_targets = {k: v[args.label_set] for k, v in X['val_targets'].items()}
        del X

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quantiles

    quantiles_train = fetch_data_as_torch(train_path, 'train_quantiles')
    quantiles_val = fetch_data_as_torch(val_path, 'val_quantiles')

    train_dataset = VgSamplerDataset(data_train, args.seq_len, device,
                                     quantiles=quantiles_train, labels=train_targets,
                                     quantile_pad_value=args.quantile_pad_value)
    val_dataset = VgSamplerDataset(data_val, args.seq_len, device,
                                   quantiles=quantiles_val, labels=val_targets,
                                   quantile_pad_value=args.quantile_pad_value)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if bool(args.test_run):
        train_loader = [X for i, X in enumerate(train_loader) if i < 2]
        val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    # unconditional label propensities

    def propensity_from(targets_dict):
        return sum(targets_dict.values()) / len(targets_dict)

    p = propensity_from(train_targets)
    print(f"Train set positive class propensity is {p}")

    if bool(args.weighted_loss):
        weights = torch.tensor([p, 1 - p]).to(device)
    else:
        weights = None

    # initialisation of model

    class BaselineNN(nn.Module):
        def __init__(self,
                     num_classes,
                     num_features,
                     hidden_dim=100,
                     weight=None,
                     with_values=True,
                     clf_dropout=0.
                     ):
            super().__init__()
            self.num_classes = num_classes
            self.num_features = num_features
            self.weight = weight.to(torch.float) if weight is not None else weight
            self.with_values = with_values
            self.clf_dropout = clf_dropout
            self.clf = nn.Sequential(
                nn.Linear(self.num_features, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(p=clf_dropout),
                nn.Linear(hidden_dim, num_classes, bias=True)
            )

        def forward(self, X, predict=False):
            if self.with_values:
                X, quantiles, targets = X
                if args.values_as == 'one-hot':
                    X = F.one_hot(X, num_tokens)
                    X = torch.flatten(X, start_dim=1)
                    quantiles = F.one_hot(quantiles, num_quantiles)
                    quantiles = torch.flatten(quantiles, start_dim=1)
                features = torch.cat([X, quantiles], dim=1).to(torch.float)
                logits = self.clf(features)
            else:
                X, targets = X
                features = X
                logits = self.clf(features)

            loss = F.cross_entropy(logits, targets, weight=self.weight)
            return logits if predict else loss

    if args.values_as == 'one-hot':
        num_features = args.seq_len * (num_tokens + num_quantiles)

    model = BaselineNN(num_classes=2,
                       num_features=num_features,
                       hidden_dim=args.clf_hidden_dim,
                       weight=weights,
                       with_values=True,
                       clf_dropout=args.clf_dropout
                       )

    model.to(device)

    # for name, param in states.named_parameters():
    #    print(name, param.requires_grad)

    print("model specification:", model, sep="\n")

    # initialise optimiser

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = model_methods.BaselineMethods(model, writer)

    # training loop

    best_val_loss = np.inf
    early_stopping_counter = 0
    for epoch in range(args.num_epochs):

        # training and evaluation

        training.train(train_loader, optimizer, epoch, grad_accum_every=args.grad_accum_every)
        val_loss = training.evaluate(val_loader, epoch)

        # tracking model classification metrics

        training.predict(train_loader, epoch, device, prefix="train")
        _, _, acc, bal_acc, roc_auc = training.predict(val_loader, epoch, device, prefix="val")

        # whether to checkpoint model

        if val_loss < best_val_loss:
            print("Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict()
            }, ckpt_path)

            print("Checkpoint saved!\n")
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter == args.early_stopping_threshold:
            print('early stopping threshold hit! ending training...')
            break

        scheduler.step()

        # flushing writer

        print(f'epoch {epoch} completed!', '\n')
        print('flushing writer...')
        writer.flush()

    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments(mode='baselining').parse()

    # check output roots exist; if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # run

    baseline(arguments)
