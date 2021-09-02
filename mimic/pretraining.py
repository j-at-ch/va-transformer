import sys
import pandas as pd
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import methods
from data_utils import *
from arguments import Arguments
from v_transformers.vtransformers import Decoder, TransformerWrapper
from v_transformers.autoregressive_wrapper import AutoregressiveWrapper


def pretrain(args):
    print('*' * 17, 'chart-transformer summoned for training with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    d_items_path = os.path.join(args.data_root, "D_LABITEMS.csv")
    train_path = os.path.join(args.data_root, "train_data.pkl")
    val_path = os.path.join(args.data_root, "val_data.pkl")
    mapping_path = os.path.join(args.data_root, "mappings.pkl")
    ckpt_path = os.path.join(args.save_root, args.model_name + ".pt")
    logs_path = os.path.join(args.logs_root, args.model_name)

    # device

    device = torch.device(args.device)

    # mappings

    mappings_dict = fetch_mappings(mapping_path)
    mappings = Mappings(mappings_dict)

    # labellers

    d_items_df = pd.read_csv(d_items_path, index_col='ITEMID', dtype={'ITEMID': str})
    labeller = Labellers(mappings_dict, d_items_df)

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

    train_dataset = VgSamplerDataset(data_train, args.seq_len, device,
                                     quantiles=quantiles_train, quantile_pad_value=args.quantile_pad_value)
    val_dataset = VgSamplerDataset(data_val, args.seq_len, device,
                                   quantiles=quantiles_val, quantile_pad_value=args.quantile_pad_value)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for rapid test run

    if bool(args.test_run):
        train_loader = [X for i, X in enumerate(train_loader) if i < 2]
        val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    # instantiate GPT-like decoder architecture

    model = TransformerWrapper(
        num_tokens=mappings.num_tokens,
        max_seq_len=args.seq_len,
        attn_layers=Decoder(
            dim=args.attn_dim,
            depth=args.attn_depth,
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            value_guided=args.value_guided,
            use_rezero=bool(args.use_rezero),
            rotary_pos_emb=bool(args.rotary_pos_emb)
        )
    )

    # wrap model for pretraining

    pre_model = AutoregressiveWrapper(model, value_guided=args.value_guided)
    pre_model.to(device)

    print("model specification:", pre_model.net, sep="\n")

    optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = methods.TrainingMethods(pre_model, writer)

    # write initial embeddings

    if bool(args.write_initial_embeddings):
        training.write_embeddings(-1, mappings, labeller, args.seq_len, device)

    # training loop

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        training.train(train_loader, optimizer, epoch)
        val_loss = training.evaluate(val_loader, epoch)

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
                training.write_embeddings(epoch, mappings, labeller, args.seq_len, device)

            print("Checkpoint saved!\n")
            best_val_loss = val_loss

        if args.value_guided[0:4] in ['vg1.']:
            training.write_g_histograms(epoch)

        print(f'epoch {epoch} completed!')
        print('flushing writer...')
        writer.flush()

        scheduler.step()

    # write final embeddings

    if bool(args.write_final_embeddings):
        training.write_embeddings(args.num_epochs, mappings, labeller, args.seq_len, device)

    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments(mode='pretraining').parse()

    # check output roots exist: if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # run pretraining

    pretrain(arguments)
