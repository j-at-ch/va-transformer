import sys
import os

import torch
import tqdm
import numpy as np
import pandas as pd
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import model_methods
from utils.data_utils import *
from utils.arguments import Arguments
from utils.mappings import Mappings, Labellers
from utils.samplers import VgSamplerDataset
from vg_transformers.vg_transformers import Decoder, TransformerWrapper
from vg_transformers.autoregressive_wrapper import AutoregressiveWrapper


def evaluate(args):
    print('*' * 17, f'vg_transformer summoned for {args.mode} with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    d_items_path = os.path.join(args.data_root, "D_LABITEMS.csv")
    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")  # improvement: pkl a class instead of dict
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)
    if args.load_from_checkpoint_at is not None:
        params_path = os.path.join(args.model_root, args.load_from_checkpoint_at)

    # device

    device = torch.device(args.device)

    # mappings

    mappings_dict = fetch_mappings(mapping_path)

    pad_token = 0
    pad_guide_token = 6
    sos_token = sos_guide_token = eos_token = eos_guide_token = None
    if bool(args.use_specials):
        sos_token, sos_guide_token = len(mappings_dict['itemid2token']), 7
        eos_token, eos_guide_token = sos_token + 1, 8

    mappings = Mappings(mappings_dict,
                        pad_token=pad_token,
                        sos_token=sos_token,
                        eos_token=eos_token,
                        pad_guide_token=pad_guide_token,
                        sos_guide_token=sos_guide_token,
                        eos_guide_token=eos_guide_token
                        )

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings, d_items_df)

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quantiles

    if args.value_guided == 'plain':
        quantiles_train = None
        quantiles_val = None
    else:
        quantiles_train = fetch_data_as_torch(train_path, 'train_quantiles')
        quantiles_val = fetch_data_as_torch(val_path, 'val_quantiles')

    # load data for pretraining based on arguments

    train_dataset = VgSamplerDataset(data_train, args.seq_len, mappings, device,
                                     quantiles=quantiles_train,
                                     use_specials=bool(args.use_specials),
                                     align_sample_at=args.align_sample_at)
    val_dataset = VgSamplerDataset(data_val, args.seq_len, mappings, device,
                                   quantiles=quantiles_val,
                                   use_specials=bool(args.use_specials),
                                   align_sample_at=args.align_sample_at)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for rapid test run

    if bool(args.test_run):
        train_loader = [X for i, X in enumerate(train_loader) if i < 2]
        val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    for i, X in enumerate(val_loader):
        print(X)

    # instantiate GPT-like decoder architecture

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        num_guide_tokens=mappings.num_guide_tokens,
        max_seq_len=args.seq_len,
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            value_guided=args.value_guided,
            dim_guide=args.attn_dim_guide,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        )
    )

    # wrap model for pretraining

    pre_model = AutoregressiveWrapper(model,
                                      value_guided=args.value_guided,
                                      ignore_index=args.ignore_index,
                                      ignore_guide_index=args.ignore_guide_index)

    if args.load_from_checkpoint_at is not None:
        checkpoint = torch.load(params_path, map_location=device)
        states = checkpoint['model_state_dict']
        pre_model.load_state_dict(states)
    else:
        pass

    pre_model.to(device)
    #print(pre_model.state_dict()['net.token_emb.weight']) sanity check
    pre_model.eval()
    with torch.no_grad():
        for i, X in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                              mininterval=0.5, desc=f'evaluation'):
            print(X)
            if i > 5:
                break
            if pre_model.value_guided[0:3] == 'vg2':
                xo = X[0][:, 1:]
                qo = X[1][:, 1:]
                out, quantiles_out = pre_model.predict(X)
                loss = pre_model(X)
                print(loss)
                print("target:",
                      xo[1, :],
                      qo[1, :],
                      "prediction:",
                      out[1, :],
                      quantiles_out[1, :],
                      sep='\n')

            elif pre_model.value_guided == 'plain':
                xi = X[:, :-1]
                xo = X[:, 1:]
                out = pre_model.predict(X)
                loss = pre_model(X)
                print("target:",
                      xo[1, :],
                      "prediction:",
                      out[1, :],
                      sep='\n')


def pretrain(args):
    print('*' * 17, 'chart-transformer summoned for training with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    d_items_path = os.path.join(args.data_root, "D_LABITEMS.csv")
    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")  # improvement: pkl a class instead of dict
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)

    # device

    device = torch.device(args.device)

    # mappings

    mappings_dict = fetch_mappings(mapping_path)

    pad_token = pad_guide_token = 0
    sos_token = sos_guide_token = eos_token = eos_guide_token = None
    if bool(args.use_specials):
        sos_token, sos_guide_token = len(mappings_dict['itemid2token']), 7
        eos_token, eos_guide_token = sos_token + 1, 8

    mappings = Mappings(mappings_dict,
                        pad_token=pad_token,
                        sos_token=sos_token,
                        eos_token=eos_token,
                        pad_guide_token=pad_guide_token,
                        sos_guide_token=sos_guide_token,
                        eos_guide_token=eos_guide_token
                        )

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings, d_items_df)

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quantiles

    if args.value_guided == 'plain':
        quantiles_train = None
        quantiles_val = None
    else:
        quantiles_train = fetch_data_as_torch(train_path, 'train_quantiles')
        quantiles_val = fetch_data_as_torch(val_path, 'val_quantiles')

    # load data for pretraining based on arguments

    train_dataset = VgSamplerDataset(data_train, args.seq_len, mappings, device,
                                     quantiles=quantiles_train,
                                     use_specials=bool(args.use_specials),
                                     align_sample_at=args.align_sample_at)
    val_dataset = VgSamplerDataset(data_val, args.seq_len, mappings, device,
                                   quantiles=quantiles_val,
                                   use_specials=bool(args.use_specials),
                                   align_sample_at=args.align_sample_at)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for rapid test run

    if bool(args.test_run):
        train_loader = [X for i, X in enumerate(train_loader) if i < 2]
        val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    # instantiate GPT-like decoder architecture

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        num_guide_tokens=mappings.num_guide_tokens,
        max_seq_len=args.seq_len,
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            value_guided=args.value_guided,
            dim_guide=args.attn_dim_guide,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        )
    )

    # wrap model for pretraining

    pre_model = AutoregressiveWrapper(model,
                                      value_guided=args.value_guided,
                                      ignore_index=args.ignore_index,
                                      ignore_guide_index=args.ignore_guide_index)
    pre_model.to(device)

    print("model specification:", pre_model.net, sep="\n")

    optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = model_methods.TrainingMethods(pre_model, writer)

    # write initial embeddings

    tokens_to_write = list(mappings.token2itemid)

    if bool(args.write_initial_embeddings):
        training.write_token_emb(-1, tokens_to_write, labeller, args.seq_len, device)

    # training loop

    best_val_loss = np.inf
    early_stopping_counter = 0
    for epoch in range(args.num_epochs):
        training.train(train_loader, optimizer, epoch,
                       grad_accum_every=args.grad_accum_every,
                       gamma=args.gamma)
        val_loss = training.evaluate(val_loader, epoch,
                                     gamma=args.gamma)

        if val_loss < best_val_loss:
            print("Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': pre_model.state_dict(),
                'args': vars(args),
                'optim_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)

            # track checkpoint's embeddings

            if bool(args.write_best_val_embeddings):
                training.write_token_emb(epoch, tokens_to_write, labeller, args.seq_len, device)

            print("Checkpoint saved!\n")
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter == args.early_stopping_threshold:
            print('early stopping threshold hit! ending training...')
            break

        print(f'epoch {epoch} completed!')
        print('flushing writer...')
        writer.flush()

        scheduler.step()

    # write final embeddings

    if bool(args.write_final_embeddings):
        training.write_token_emb(args.num_epochs, tokens_to_write, labeller, args.seq_len, device)

    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments(mode='pretraining').parse()

    # check output roots exist: if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # check gamma makes sense

    assert (arguments.gamma >= 0) and (arguments.gamma <= 1), "--gamma should satisfy 0 <= gamma <= 1."

    # run pretraining
    print(f"mode is {arguments.mode}")

    if arguments.mode == 'train':
        pretrain(arguments)
    elif arguments.mode == 'evaluate':
        evaluate(arguments)

