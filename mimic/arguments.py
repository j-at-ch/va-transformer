import os
import argparse
from pprint import pprint


class Arguments:
    def __init__(self, mode):
        self.parser = argparse.ArgumentParser(description='chart-transformer')
        self.mode = mode
        self.arguments = None

    def initialise(self):

        # data roots

        self.parser.add_argument('--mimic_root', type=str)
        self.parser.set_defaults(mimic_root='/home/james/Documents/Charters/all_rows/data')
        self.parser.add_argument('--data_root', type=str)
        self.parser.set_defaults(data_root='/home/james/Documents/Charters/all_rows/data')
        self.parser.add_argument('--model_root', type=str)
        self.parser.set_defaults(model_root='/home/james/Documents/Charters/all_rows/models')
        self.parser.add_argument('--save_root', type=str)
        self.parser.set_defaults(save_root='/home/james/Documents/Charters/all_rows/results')
        self.parser.add_argument('--logs_root', type=str)
        self.parser.set_defaults(logs_root='/home/james/Documents/Charters/all_rows/logs')

        # pretraining constants

        self.parser.add_argument('--num_epochs', type=int, default=3)
        self.parser.add_argument('--batch_size_tr', type=int, default=100)
        self.parser.add_argument('--batch_size_val', type=int, default=100)
        self.parser.add_argument('--validate_every', type=int, default=10)  # note: deprecated
        self.parser.add_argument('--checkpoint_after', type=int, default=100)
        self.parser.add_argument('--generate_every', type=int, default=20)  # note: deprecated
        self.parser.add_argument('--generate_length', type=int, default=200)  # note: deprecated
        self.parser.add_argument('--seq_len', type=int, default=200)  #

        # attention specification

        self.parser.add_argument('--attn_dim', type=int, default=100)
        self.parser.add_argument('--attn_depth', type=int, default=3)
        self.parser.add_argument('--attn_heads', type=int, default=4)
        self.parser.add_argument('--attn_dropout', type=float, default=0.)
        self.parser.add_argument('--ff_dropout', type=float, default=0.)

        # general arguments

        self.parser.add_argument('--model_name', type=str, default='test_experiment')
        self.parser.add_argument('--writer_flush_secs', type=int, default=120)
        self.parser.add_argument('--write_best_val_embeddings', type=bool, default=False)
        self.parser.add_argument('--write_initial_embeddings', type=bool, default=True)
        self.parser.add_argument('--write_final_embeddings', type=bool, default=True)
        self.parser.add_argument('--device', type=str, default="cuda:0")
        self.parser.add_argument('--learning_rate', type=float, default=1e-4)
        self.parser.add_argument('--scheduler_decay', type=float, default=1)
        self.parser.add_argument('--test_run', type=bool, default=False)

        # finetuning arguments

        if self.mode == 'finetuning':
            self.parser.add_argument('--ft_batch_size', type=int, default=100)
            self.parser.add_argument('--label_set', type=str, required=True)
            self.parser.add_argument('--pretuned_model', type=str, required=True)
            self.parser.add_argument('--weighted_loss', type=bool, default=False)

    def parse(self, verbose=False):
        self.initialise()
        self.arguments = self.parser.parse_args()
        if verbose: pprint(vars(self.arguments), indent=4)
        return self.arguments


class PreprocessingArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='preprocessor')
        self.arguments = None

    def initialise(self):

        # data roots

        self.parser.add_argument('--mimic_root', type=str)
        self.parser.set_defaults(mimic_root='/home/james/Documents/Charters/mimic-iii-clinical-database-1.4')
        self.parser.add_argument('--save_root', type=str)
        self.parser.set_defaults(save_root='/home/james/Documents/Charters/preprocessing_output')

        # general arguments

        ####

    def parse(self, verbose=False):
        self.initialise()
        self.arguments = self.parser.parse_args()
        if verbose: pprint(vars(self.arguments), indent=4)
        return self.arguments
