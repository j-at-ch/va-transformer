import os

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
from utils.samplers import VgSamplerDataset, cycler
from vg_transformers.va_transformers import Decoder, TransformerWrapper
from vg_transformers.autoregressive_wrapper import AutoregressiveWrapper


def main(args):
    print('*' * 17, f'vg-transformer summoned for {args.mode} with the following settings:', sep='\n')
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

    pad_token = args.pad_token
    pad_quant_token = args.pad_quant_token
    sos_token = sos_quant_token = eos_token = eos_quant_token = None
    if args.specials == 'SOS':
        sos_token, sos_quant_token = len(mappings_dict['itemid2token']), 7
    elif args.specials == 'EOS':
        eos_token, eos_quant_token = len(mappings_dict['itemid2token']), 7
    elif args.specials == 'both':
        sos_token, sos_quant_token = len(mappings_dict['itemid2token']), 7
        eos_token, eos_quant_token = len(mappings_dict['itemid2token']) + 1, 8

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

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings, d_items_df)

    # get tokens

    data_train = fetch_data_as_torch(train_path, 'train_tokens')
    data_val = fetch_data_as_torch(val_path, 'val_tokens')

    # get quantiles

    if bool(args.with_values):
        quants_train = fetch_data_as_torch(train_path, 'train_quantiles')
        quants_val = fetch_data_as_torch(val_path, 'val_quantiles')
    else:
        quants_train = None
        quants_val = None

    # load data for pretraining based on arguments

    train_dataset = VgSamplerDataset(data_train, args.seq_len, mappings, device,
                                     quants=quants_train,
                                     specials=args.specials,
                                     align_sample_at=args.align_sample_at)
    val_dataset = VgSamplerDataset(data_val, args.seq_len, mappings, device,
                                   quants=quants_val,
                                   specials=args.specials,
                                   align_sample_at=args.align_sample_at)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    if bool(args.test_run):
        train_loader = make_toy_loader(train_loader)
        val_loader = make_toy_loader(val_loader)

    # instantiate GPT-like decoder architecture

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        num_quant_tokens=mappings.num_quant_tokens,
        max_seq_len=args.seq_len,
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            quant_guides=args.quant_guides,
            dim_quants=args.attn_dim_quants,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        ),
        use_quant_pos_emb=bool(args.use_quant_pos_emb),
        conditional_logit=args.conditional_logit,
        va_transformer=bool(args.va_transformer)
    )

    # wrap model for pretraining

    pre_model = AutoregressiveWrapper(model,
                                      quant_guides=args.quant_guides,
                                      ignore_index=args.ignore_index,
                                      ignore_quant_index=args.ignore_quant_index)

    if args.load_from_checkpoint_at is not None:
        params_path = os.path.join(args.model_root, args.load_from_checkpoint_at)
        checkpoint = torch.load(params_path, map_location=device)
        states = checkpoint['model_state_dict']
        pre_model.load_state_dict(states)
    else:
        pass

    pre_model.to(device)

    print("model specification:", pre_model.net, sep="\n")

    if args.mode == 'pretraining':

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
            training.train(train_loader, optimizer, epoch, grad_accum_every=args.grad_accum_every, gamma=args.gamma)
            val_losses = training.evaluate(val_loader, epoch, gamma=args.gamma)
            if val_losses.loss < best_val_loss:
                print("Saving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': pre_model.state_dict(),
                    'args': vars(args),
                    'optim_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses.loss,
                    'token_loss': val_losses.token_loss,
                    'quantile_loss': val_losses.quantile_loss
                }, ckpt_path)

                # track checkpoint's embeddings

                if bool(args.write_best_val_embeddings):
                    training.write_token_emb(epoch, tokens_to_write, labeller, args.seq_len, device)

                print("Checkpoint saved!\n")
                best_val_loss = val_losses.loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print(f'epoch {epoch} completed!')
            print('flushing writer...')
            writer.flush()

            if early_stopping_counter == args.early_stopping_threshold:
                print('early stopping threshold hit! ending training...')
                break

            scheduler.step()

        # write final embeddings

        if bool(args.write_final_embeddings):
            training.write_token_emb(epoch, tokens_to_write, labeller, args.seq_len, device)

        writer.close()
        print("training finished and writer closed!")

    elif args.mode == 'evaluation':
        pre_model.eval()

        X = next(cycler(val_loader))
        if pre_model.quant_guides is None:
            xi = X[:, :-1]
            xo = X[:, 1:]
            out = pre_model.predict(X)
            loss = pre_model(X)
            print(loss,
                  "target_tokens:",
                  xo[1, :],
                  "predicted_tokens:",
                  out[1, :],
                  sep='\n')
        else:
            xo = X[0][:, 1:]
            qo = X[1][:, 1:]
            out, quantiles_out = pre_model.predict(X)
            loss = pre_model(X)
            print(loss,
                  "target_tokens:",
                  xo[1, :],
                  "predicted_tokens:",
                  out[1, :],
                  "target_quantiles:",
                  qo[1, :],
                  "predicted_quantiles:",
                  quantiles_out[1, :],
                  sep='\n')


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
    main(arguments)


