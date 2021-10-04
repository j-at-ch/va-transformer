import os
import numpy as np
from pprint import pprint
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import model_methods
from utils.data_utils import *
from utils.arguments import Arguments
from utils.mappings import Mappings
from utils.samplers import SeqSamplerDataset
from va_transformers.finetuning_wrapper import Classifier, SimpleClassifier


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

    train_tgt_path = os.path.join(args.data_root, "train_targets.pkl")
    val_tgt_path = os.path.join(args.data_root, "val_targets.pkl")

    # device

    device = torch.device(args.device)

    # mappings

    mappings_dict = fetch_mappings(mapping_path)
    len_t_dict = len(mappings_dict['itemid2token'])
    len_q_dict = len(mappings_dict['qname2qtoken'])

    pad_token = args.pad_token
    pad_quant_token = args.pad_quant_token if args.with_values else None
    sos_token = sos_quant_token = eos_token = eos_quant_token = None

    if args.specials == 'SOS':
        sos_token = len_t_dict
        sos_quant_token = len_q_dict if args.with_values else None
    elif args.specials == 'EOS':
        eos_token = len_t_dict
        eos_quant_token = len_q_dict if args.with_values else None
    elif args.specials == 'both':
        sos_token = len_t_dict
        sos_quant_token = len_q_dict if args.with_values else None
        eos_token = len_t_dict + 1
        eos_quant_token = (len_q_dict + 1) if args.with_values else None

    mappings = Mappings(mappings_dict,
                        pad_token=pad_token,
                        sos_token=sos_token,
                        eos_token=eos_token,
                        pad_quant_token=pad_quant_token,
                        sos_quant_token=sos_quant_token,
                        eos_quant_token=eos_quant_token
                        )

    print(f"[PAD] token is {mappings.pad_token}",
          f"[SOS] token is {mappings.sos_token}",
          f"[EOS] token is {mappings.eos_token}",
          f"[PAD] quant token is {mappings.pad_quant_token}",
          f"[SOS] quant token is {mappings.sos_quant_token}",
          f"[EOS] quant token is {mappings.eos_quant_token}",
          sep="\n")

    # fetch labels

    with open(train_tgt_path, 'rb') as f:
        X = pickle.load(f)
        train_targets = {k: v[args.targets] for k, v in X['train_targets'].items()}
        del X

    with open(val_tgt_path, 'rb') as f:
        X = pickle.load(f)
        val_targets = {k: v[args.targets] for k, v in X['val_targets'].items()}
        del X

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quants

    if not bool(args.with_values):
        quants_train = None
        quants_val = None
    else:
        quants_train = fetch_data_as_torch(train_path, 'train_quants')
        quants_val = fetch_data_as_torch(val_path, 'val_quants')

    train_dataset = SeqSamplerDataset(data_train, args.seq_len, mappings, device,
                                      quants=quants_train,
                                      targets=train_targets,
                                      specials=args.specials,
                                      align_sample_at=args.align_sample_at)
    val_dataset = SeqSamplerDataset(data_val, args.seq_len, mappings, device,
                                    quants=quants_val,
                                    targets=val_targets,
                                    specials=args.specials,
                                    align_sample_at=args.align_sample_at)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if bool(args.toy_run):
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

    class BaseliningWrapper(nn.Module):
        def __init__(self,
                     num_classes,
                     num_features,
                     clf_or_reg='clf',
                     hidden_dim=100,
                     weight=None,
                     with_values=True,
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
            self.with_values = with_values
            self.clf = Classifier(num_features, hidden_dim, num_classes, clf_dropout) if clf_depth == 2 \
                else SimpleClassifier(num_features, num_classes, clf_dropout)

        def forward(self, x, predict=False):
            if self.clf_or_reg == 'reg':
                assert self.num_classes == 1, "if in regression mode, num_classes must be 1"

            if self.with_values:
                x, quants, targets = x
                targets = targets.long() if self.clf_or_reg == 'clf' else targets.float()
                if args.values_as == 'one-hot':
                    x = F.one_hot(x, mappings.num_tokens)
                    x = torch.flatten(x, start_dim=1)
                    quants = F.one_hot(quants, mappings.num_quant_tokens)
                    quants = torch.flatten(quants, start_dim=1)
                features = torch.cat([x, quants], dim=1).float()
            else:
                x, targets = x
                if args.values_as == 'one-hot':
                    x = F.one_hot(x, mappings.num_tokens)
                    x = torch.flatten(x, start_dim=1)
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

    if args.with_values:
        if args.values_as == 'one-hot':
            num_features = args.seq_len * (mappings.num_tokens + mappings.num_quant_tokens)
        else:
            num_features = args.seq_len * 2
    else:
        if args.values_as == 'one-hot':
            num_features = args.seq_len * mappings.num_tokens
        else:
            num_features = args.seq_len

    model = BaseliningWrapper(num_classes=args.num_classes,
                              num_features=num_features,
                              clf_or_reg=args.clf_or_reg,
                              hidden_dim=args.clf_hidden_dim,
                              weight=weights,
                              with_values=bool(args.with_values),
                              clf_dropout=args.clf_dropout,
                              clf_depth=args.clf_depth
                              )
    model.to(device)

    print("model specification:", model, sep="\n")

    if args.mode == "training":

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
        writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
        training = model_methods.BaselineMethods(model, writer, clf_or_reg=args.clf_or_reg)

        # training loop

        best_val_loss = np.inf
        early_stopping_counter = 0
        for epoch in range(args.num_epochs):

            # training and evaluation

            training.train(train_loader, optimizer, epoch, grad_accum_every=args.grad_accum_every)
            val_loss = training.evaluate(val_loader, epoch)
            _, _, metrics = training.predict(val_loader, epoch, device, prefix="val")

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

    if bool(args.WARNING_TESTING):
        print("\nWARNING TEST set in use!\n")

        # load test set data
        test_path = os.path.join(args.data_root, "test_data.pkl")
        data_test = fetch_data_as_torch(test_path, 'test_tokens')
        if bool(args.with_values):
            quants_test = fetch_data_as_torch(test_path, 'test_quants')
        else:
            quants_test = None
        test_tgt_path = os.path.join(args.data_root, "test_targets.pkl")
        with open(test_tgt_path, 'rb') as f:
            x = pickle.load(f)
            test_targets = {k: v[args.targets] for k, v in x['test_targets'].items()}
            del x

        test_dataset = SeqSamplerDataset(data_test, args.seq_len, mappings, device,
                                         quants=quants_test,
                                         targets=test_targets,
                                         specials=args.specials,
                                         align_sample_at=args.align_sample_at)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size_tr, shuffle=True)

        if bool(args.toy_run):
            test_loader = make_toy_loader(test_loader)

        # test the model at the checkpoint
        params_path = os.path.join(args.model_root, args.model_name + '.pt')
        checkpoint = torch.load(params_path, map_location=device)
        states = checkpoint['model_state_dict']
        model.load_state_dict(states)

        model.to(device)
        writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)

        testing = model_methods.BaselineMethods(model, writer=writer, clf_or_reg=args.clf_or_reg)
        val_losses = testing.evaluate(val_loader, 0, prefix='re-val')
        _, _, val_metrics = testing.predict(val_loader, 0, device, prefix='re-val')
        test_losses = testing.evaluate(test_loader, 0, prefix='test')
        _, _, test_metrics = testing.predict(test_loader, 0, device, prefix='test')

        writer.flush()
        writer.close()
        print("testing finished and writer closed!")

        # write results to auxiliary logs file for convenience

        print("writing baselining logs to central csv for convenience!")
        central_logs_name = f'baselining_{args.targets}_logs.csv'
        central_logs_path = os.path.join(args.logs_root, central_logs_name)
        if not os.path.isfile(central_logs_path):
            with open(central_logs_path, 'w') as f:
                if args.clf_or_reg == "clf":
                    f.write(f"model_name,"
                            f"val_loss,test_loss,bal_acc_val,bal_acc_tst,roc_val,roc_tst\n")
                else:
                    f.write(f"model_name,"
                            f"val_loss,test_loss,mse_val,mse_tst,r2_val,r2_tst\n")
        with open(central_logs_path, 'a') as f:
            if args.clf_or_reg == "clf":
                f.write(f"{args.model_name},{val_losses:.4f},{test_losses:.4f}"
                        f",{val_metrics['bal_acc']:.4f},{test_metrics['bal_acc']:.4f}"
                        f",{val_metrics['roc_auc']:.4f},{test_metrics['roc_auc']:.4f}\n")
            else:
                f.write(f"{args.model_name},{val_losses:.4f},{test_losses:.4f},"
                        f"{val_metrics['mse']:.4f},{test_metrics['mse']:.4f},"
                        f"{val_metrics['r2']:.4f},{test_metrics['r2']:.4f}\n")
        print(f"metrics written to {central_logs_path}")


if __name__ == "__main__":
    arguments = Arguments(mode='baselining').parse()

    # check output roots exist; if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # run
    baseline(arguments)
