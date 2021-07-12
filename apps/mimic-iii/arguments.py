import argparse
from pprint import pprint


class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='chart-transformer')
        self.arguments = None

    def initialise(self):
        # data roots

        self.parser.add_argument('--mimic_root', type=str,
                                 default='C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4', help='')
        self.parser.add_argument('--data_root', type=str,
                                 default='C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers', help='')
        self.parser.add_argument('--save_root', type=str,
                                 default='C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers', help='')

        # constants

        self.parser.add_argument('--NUM_EPOCHS', type=int, default=10)
        self.parser.add_argument('--NUM_BATCHES', type=int, default=100)
        self.parser.add_argument('--BATCH_SIZE', type=int, default=4)
        self.parser.add_argument('--GRADIENT_ACCUMULATE_EVERY', type=int, default=4)
        self.parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
        self.parser.add_argument('--VALIDATE_EVERY', type=int, default=10)
        self.parser.add_argument('--CHECKPOINT_AFTER', type=int, default=10)
        self.parser.add_argument('--GENERATE_EVERY', type=int, default=20)
        self.parser.add_argument('--GENERATE_LENGTH', type=int, default=200)
        self.parser.add_argument('--SEQ_LEN', type=int, default=200)

        # model specification

        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='# layers in linear model')

        # training specification

        self.parser.add_argument('--lr', type=float)

        # set defaults

        self.parser.set_defaults(max_norm=True)

    def parse(self, verbose=False):
        self.initialise()
        self.arguments = self.parser.parse_args()
        if verbose: pprint(vars(self.arguments), indent=2)
        return self.arguments
