import os
import argparse
from pprint import pprint


class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='chart-transformer')
        self.arguments = None

    def initialise(self):

        # data roots

        self.parser.add_argument('--mimic_root', type=str)
        self.parser.set_defaults(mimic_root='C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4')
        self.parser.add_argument('--data_root', type=str)
        self.parser.set_defaults(data_root='C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers')
        self.parser.add_argument('--save_root', type=str)
        self.parser.set_defaults(save_root='C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers')
        self.parser.add_argument('--logs_root', type=str)
        self.parser.set_defaults(logs_root='C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers')

        # constants

        self.parser.add_argument('--num_epochs', type=int, default=10)
        self.parser.add_argument('--num_batches_tr', type=int, default=1000)
        self.parser.add_argument('--num_batches_val', type=int, default=1000)
        self.parser.add_argument('--batch_size_tr', type=int, default=4)
        self.parser.add_argument('--batch_size_val', type=int, default=4)
        self.parser.add_argument('--grad_accumulate_every', type=int, default=4)
        self.parser.add_argument('--learning_rate', type=float, default=1e-4)
        self.parser.add_argument('--validate_every', type=int, default=10)
        self.parser.add_argument('--checkpoint_after', type=int, default=100)
        self.parser.add_argument('--generate_every', type=int, default=20)
        self.parser.add_argument('--generate_length', type=int, default=200)
        self.parser.add_argument('--seq_len', type=int, default=200)

        # model specification

        self.parser.add_argument('--attn_dim', type=int, default=100)
        self.parser.add_argument('--attn_depth', type=int, default=3)
        self.parser.add_argument('--attn_heads', type=int, default=4)

    def parse(self, verbose=False):
        self.initialise()
        self.arguments = self.parser.parse_args()
        if verbose: pprint(vars(self.arguments), indent=4)
        return self.arguments
