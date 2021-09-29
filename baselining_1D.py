import os
import numpy as np
from pprint import pprint

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import model_methods
from utils.data_utils import *
from utils.arguments import Arguments
from utils.mappings import Mappings
from utils.samplers import SeqSamplerDataset, V1dDataset
from va_transformers.finetuning_wrapper import Classifier, SimpleClassifier


def baseline_for_1D(args):
    print('*' * 17, 'summoning baseline models for classification with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    train_path = os.path.join(args.data_root, "train_data1D.pkl")
    val_path = os.path.join(args.data_root, "val_data1D.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)
    mapping_path = os.path.join(args.data_root, "mappings.pkl")

    train_tgt_path = os.path.join(args.data_root, "train_targets.pkl")
    val_tgt_path = os.path.join(args.data_root, "val_targets.pkl")

    # device

    device = torch.device(args.device)

    # mappings

    mappings_dict = fetch_mappings(mapping_path)
    mappings = Mappings(mappings_dict)

    # fetch targets

    with open(train_tgt_path, 'rb') as f:
        X = pickle.load(f)
        train_targets = {k: v[args.targets] for k, v in X['train_targets'].items()}
        del X

    with open(val_tgt_path, 'rb') as f:
        X = pickle.load(f)
        val_targets = {k: v[args.targets] for k, v in X['val_targets'].items()}
        del X

    # get data

    train_data = fetch_data_as_torch(train_path, 'train_values_mean')
    val_data = fetch_data_as_torch(val_path, 'val_values_mean')

    # get quantiles

    train_dataset = V1dDataset(train_data, mappings, device, targets=train_targets)
    val_dataset = V1dDataset(val_data, mappings, device, targets=val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if bool(args.test_run):
        train_loader = make_toy_loader(train_loader)
        val_loader = make_toy_loader(val_loader)

    # weighting with target propensities if necessary

    weights = None

    if args.clf_or_reg == 'clf':
        def propensity_from(targets_dict):
            return sum(targets_dict.values()) / len(targets_dict)

        p = propensity_from(train_targets)
        print(f"Train set positive class propensity is {p}")

        if bool(args.weighted_loss):
            weights = torch.tensor([p, 1 - p]).to(device)

    # initialisation of model

    class Baseline1dWrapper(nn.Module):
        def __init__(self,
                     num_classes,
                     num_features,
                     clf_or_reg='clf',
                     hidden_dim=100,
                     weight=None,
                     clf_dropout=0.,
                     clf_depth=2
                     ):
            super().__init__()
            self.num_classes = num_classes
            self.num_features = num_features
            self.clf_or_reg = clf_or_reg
            self.weight = weight.to(torch.float) if weight is not None else weight
            self.clf_dropout = clf_dropout
            self.clf_depth = clf_depth
            self.clf = Classifier(num_features, hidden_dim, num_classes, clf_dropout) if clf_depth == 2 \
                else SimpleClassifier(num_features, num_classes, clf_dropout)

        def forward(self, x, predict=False):
            if self.clf_or_reg == 'reg':
                assert self.num_classes == 1, "if in regression mode, num_classes must be 1"

            x, targets = x
            targets = targets.long() if self.clf_or_reg == 'clf' else targets.float()
            features = x.float()

            if self.clf_or_reg == 'reg':
                pre_act = torch.squeeze(self.clf(features))
                preds = F.softplus(pre_act)
                loss = F.mse_loss(preds, targets)
                return preds if predict else loss

            logits = self.clf(features)
            loss = F.cross_entropy(logits, targets, weight=self.weight)
            return logits if predict else loss

    model = Baseline1dWrapper(num_classes=args.num_classes,
                              num_features=mappings.num_tokens,
                              clf_or_reg=args.clf_or_reg,
                              hidden_dim=args.clf_hidden_dim,
                              weight=weights,
                              clf_dropout=args.clf_dropout,
                              clf_depth=args.clf_depth
                              )
    model.to(device)

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

        if args.clf_or_reg == 'clf':
            _, _, metrics = training.predict(val_loader, epoch, device, prefix="val", clf_or_reg=args.clf_or_reg)
        elif args.clf_or_reg == 'reg':
            _, _, metrics = training.predict(val_loader, epoch, device, prefix="val", clf_or_reg=args.clf_or_reg)

        # whether to checkpoint model

        if val_loss < best_val_loss:
            print("Saving checkpoint because best val_loss attained...")
            torch.save({
                'epoch': epoch,
                'val_loss': val_loss,
                'metrics': metrics,
                'args': vars(args),
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict()
            }, ckpt_path)

            print("Checkpoint saved!\n")
            best_val_loss = min(val_loss, best_val_loss)
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
    baseline_for_1D(arguments)
