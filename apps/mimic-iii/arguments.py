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

        # pretraining constants

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

        # attention specification

        self.parser.add_argument('--attn_dim', type=int, default=100)
        self.parser.add_argument('--attn_depth', type=int, default=3)
        self.parser.add_argument('--attn_heads', type=int, default=4)

        # pretraining specs

        self.parser.add_argument('--model_name', type=str, default='model_exp')
        self.parser.add_argument('--writer_flush_secs', type=int, default=120)

        # finetuning arguments

        self.parser.add_argument('--ft_batch_size', type=int, default=100)
        self.parser.add_argument('--label_set', type=str, default='readm_30', choices=['readm_30', 'readm_7'])
        self.parser.add_argument('--pretuned_model', type=str, default='')

    def parse(self, verbose=False):
        self.initialise()
        self.arguments = self.parser.parse_args()
        if verbose: pprint(vars(self.arguments), indent=4)
        return self.arguments
