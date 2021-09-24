import os
import argparse


class PreprocessingArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='preprocessor')
        self.arguments = None

    def initialise(self):
        self.parser.add_argument('--mimic_root', type=str)
        self.parser.set_defaults(mimic_root='/home/james/Documents/Charters/mimic-iii-clinical-database-1.4')
        self.parser.add_argument('--save_root', type=str)
        self.parser.set_defaults(save_root='/home/james/Documents/Charters/preprocessing_output')
        self.parser.add_argument('--min_num_labs', type=int, default=1)

    def parse(self):
        self.initialise()
        self.arguments = self.parser.parse_args()
        return self.arguments
