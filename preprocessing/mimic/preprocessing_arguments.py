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
        self.parser.add_argument('--data_root', type=str)
        self.parser.set_defaults(data_root='/home/james/Documents/Charters/labs_dataset5/data')
        self.parser.add_argument('--min_num_labs', type=int, default=10)
        self.parser.add_argument('--augmented_admissions', type=str, default="r", choices=["r", "w"])
        self.parser.add_argument('--quantiles', type=list, default=[0.1, 0.25, 0.75, 0.9])
        self.parser.add_argument('--generating_data_for', type=str, default='1.5D')

    def parse(self):
        self.initialise()
        self.arguments = self.parser.parse_args()
        return self.arguments
