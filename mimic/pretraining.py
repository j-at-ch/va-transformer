import pandas as pd
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

# repo imports

import methods
from data_utils import *
from arguments import Arguments
from z_transformers.transformers import Decoder, TransformerWrapper
from z_transformers.autoregressive_wrapper import AutoregressiveWrapper


def pretrain(args):
    print('*'*17, 'chart-transformer summoned for training with the following settings:', sep='\n')
    pprint(vars(args), indent=2)

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
            heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout
        )
    )

    # wrap for autoregressive

    pre_model = AutoregressiveWrapper(model)
    pre_model.to(device)

    # load data for pretraining based on arguments

    train_dataset = ClsSamplerDataset(data_train, args.seq_len, device)
    val_dataset = ClsSamplerDataset(data_val, args.seq_len, device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_tr, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)

    #  for quick test run

    if args.test_run:
        train_loader = [X for i, X in enumerate(train_loader) if i < 2]
        val_loader = [X for i, X in enumerate(val_loader) if i < 2]

    train_cycler = cycle(train_loader)
    val_cycler = cycle(val_loader)

    optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_decay)
    writer = SummaryWriter(log_dir=logs_path, flush_secs=args.writer_flush_secs)
    training = methods.TrainingMethods(pre_model, writer)

    # write initial embeddings

    if args.write_initial_embeddings:
        print("Writing initial token embeddings to writer...")
        training.write_embeddings(0, mappings, labeller, args.seq_len, device)
        print("Initial token embeddings written!")

    # training loop

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        ________ = training.train(train_loader, optimizer, epoch)
        val_loss = training.evaluate(val_loader, epoch)

        if val_loss < best_val_loss:
            print("Saving checkpoint...")
            torch.save({
                'train_epoch': epoch,
                'model_state_dict': pre_model.state_dict(),
                'args': vars(args),
                'seq_len': args.seq_len,
                'optim_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)

            # track checkpoint's embeddings
            if args.write_embeddings & (epoch > int(args.num_epochs/2)):
                print("Writing checkpoint's token embeddings to writer...")
                training.write_embeddings(epoch + 1, mappings, labeller, args.seq_len, device)
                print("Checkpoint's token embeddings written!")

            print("Checkpoint saved!\n")
            best_val_loss = val_loss

        print(f'epoch {epoch} completed!')
        print('flushing writer...')
        writer.flush()

        # update scheduler

        scheduler.step()

    writer.close()
    print("training finished and writer closed!")


if __name__ == "__main__":
    arguments = Arguments(mode='pretraining').parse()

    # check output roots exist; if not, create...

    if not os.path.exists(arguments.save_root):
        os.mkdir(arguments.save_root)
    if not os.path.exists(arguments.logs_root):
        os.mkdir(arguments.logs_root)

    # run pretraining

    pretrain(arguments)
