import argparse


class PreprocessingArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='preprocessor')
        self.arguments = None

    def initialise(self):
        self.parser.add_argument('--mimic_root', type=str)
        self.parser.set_defaults(mimic_root=None)
        self.parser.add_argument('--save_root', type=str)
        self.parser.set_defaults(save_root="/home/james/Documents/Charters/preprocessing_output")
        self.parser.add_argument('--data_root', type=str)
        self.parser.set_defaults(data_root="/home/james/Documents/Charters/labs_dataset6/data")
        self.parser.add_argument('--min_num_labs', type=int, default=10)
        self.parser.add_argument('--augmented_admissions', type=str, default="w", choices=["r", "w"])
        self.parser.add_argument('--quantiles', type=list, default=[0.1, 0.25, 0.75, 0.9])
        self.parser.add_argument('--labs_preliminaries_done', type=int, default=0)
        self.parser.add_argument('--write_scaled_labs', type=int, default=1)
        self.parser.add_argument('--write_quantiles_summary', type=int, default=1)
        self.parser.add_argument('--preprocess_for', type=str, default='1.5D', choices=['1D', '1.5D'])
        self.parser.add_argument('--pad_mean', type=float, default=0)
        self.parser.add_argument('--pad_count', type=float, default=0)
        self.parser.add_argument('--pad_latest', type=float, default=0)
        self.parser.add_argument('--sentinel_cat', type=float, default=1)
        self.parser.add_argument('--pad_quant', type=float, default=0)

    def parse(self):
        self.initialise()
        self.arguments = self.parser.parse_args()
        return self.arguments
