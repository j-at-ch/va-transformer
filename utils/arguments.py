import os
import argparse


class Arguments:
    def __init__(self, mode):
        self.parser = argparse.ArgumentParser(description='chart-transformer')
        self.mode = mode
        self.arguments = None

    def initialise(self):

        # data roots

        self.parser.add_argument('--data_root', type=str)
        self.parser.set_defaults(data_root='/home/james/Documents/Charters/labs_dataset4/data')
        self.parser.add_argument('--model_root', type=str)
        self.parser.set_defaults(model_root='/home/james/Documents/Charters/labs_dataset4/models')
        self.parser.add_argument('--save_root', type=str)
        self.parser.set_defaults(save_root='/home/james/Documents/Charters/labs_dataset4/models')
        self.parser.add_argument('--logs_root', type=str)
        self.parser.set_defaults(logs_root='/home/james/Documents/Charters/labs_dataset4/logs')

        # use value-guided and define appropriate type-checking

        def none_or_str(value):
            if value == 'None':
                return None
            return value

        self.parser.add_argument('--value_guides', type=none_or_str, default=None,
                                 choices=[None, 'no-mixing', 'g-on-t-dev', 'g-on-t', 't-on-g', 'g-and-t'])

        # attention specification

        self.parser.add_argument('--attn_dim', type=int, default=100)
        self.parser.add_argument('--attn_depth', type=int, default=4)
        self.parser.add_argument('--attn_heads', type=int, default=8)
        self.parser.add_argument('--attn_dim_guide', type=int, default=10)
        self.parser.add_argument('--attn_dropout', type=float, default=0.05)
        self.parser.add_argument('--ff_dropout', type=float, default=0.05)
        self.parser.add_argument('--use_rezero', type=int, default=0)
        self.parser.add_argument('--rotary_pos_emb', type=int, default=0)
        self.parser.add_argument('--use_guide_pos_emb', type=int, default=0)

        # general arguments

        self.parser.add_argument('--model_name', type=str, default=f'{self.mode}_test')
        self.parser.add_argument('--num_epochs', type=int, default=50)
        self.parser.add_argument('--batch_size_tr', type=int, default=100)
        self.parser.add_argument('--batch_size_val', type=int, default=100)
        self.parser.add_argument('--only_checkpoint_after', type=int, default=100)
        # self.parser.add_argument('--generate_every', type=int, default=20)
        # self.parser.add_argument('--generate_length', type=int, default=200)
        self.parser.add_argument('--seq_len', type=int, default=200)
        self.parser.add_argument('--writer_flush_secs', type=int, default=120)
        self.parser.add_argument('--write_best_val_embeddings', type=int, default=0)
        self.parser.add_argument('--write_initial_embeddings', type=int, default=0)
        self.parser.add_argument('--write_final_embeddings', type=int, default=0)
        self.parser.add_argument('--device', type=str, default="cuda:0")
        self.parser.add_argument('--learning_rate', type=float, default=1e-4)
        self.parser.add_argument('--scheduler_decay', type=float, default=1.)
        self.parser.add_argument('--test_run', type=int, default=0)
        self.parser.add_argument('--pad_token', type=int, default=0)
        self.parser.add_argument('--pad_guide_token', type=int, default=6)
        self.parser.add_argument('--ignore_index', type=int, default=-100)
        self.parser.add_argument('--ignore_guide_index', type=int, default=-100)
        self.parser.add_argument('--grad_accum_every', type=int, default=1)
        self.parser.add_argument('--early_stopping_threshold', type=int, default=5)
        self.parser.add_argument('--gamma', type=float, default=0.5)
        # self.parser.add_argument('--use_specials', type=int, default=0) deprecated
        self.parser.add_argument('--specials', type=none_or_str, default='EOS')
        self.parser.add_argument('--align_sample_at', type=str, default='random/SOS',
                                 choices=['SOS', 'EOS', 'random/SOS', 'random/EOS'])
        if self.mode == 'pretraining':
            self.parser.add_argument('--mode', type=str, default='pretraining',
                                     choices=['pretraining', 'evaluation'])
            self.parser.add_argument('--load_from_checkpoint_at', type=str, default=None)

        # finetuning/baselining arguments

        if self.mode == 'finetuning':
            self.parser.add_argument('--mode', type=str, default='training')
            self.parser.add_argument('--pretrained_model', type=str, required=True)
            self.parser.add_argument('--load_from', type=str, default='pretrained')
            self.parser.add_argument('--targets', type=str, required=True)
            self.parser.add_argument('--weighted_loss', type=int, default=1)
            self.parser.add_argument('--freeze_base', type=int, default=0)
            self.parser.add_argument('--clf_style', type=str, default='flatten',
                                     choices=['flatten', 'sum', 'on_SOS', 'on_EOS', 'on_EOS_token', 'on_EOS-2_tokens'])
            self.parser.add_argument('--clf_hidden_dim', type=int, default=100)
            self.parser.add_argument('--clf_dropout', type=float, default=0.)
            self.parser.add_argument('--clf_or_reg', type=str, default='clf', choices=['reg', 'clf'])
            self.parser.add_argument('--predict_on_train', type=int, default=0)
        elif self.mode == 'baselining':
            self.parser.add_argument('--targets', type=str, required=True)
            self.parser.add_argument('--values_as', type=str, default='one-hot', choices=['int', 'one-hot'])
            self.parser.add_argument('--weighted_loss', type=int, default=1)
            self.parser.add_argument('--clf_dropout', type=float, default=0.)
            self.parser.add_argument('--clf_hidden_dim', type=int, default=100)

    def parse(self):
        self.initialise()
        self.arguments = self.parser.parse_args()
        return self.arguments


class PreprocessingArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='preprocessor')
        self.arguments = None

    def initialise(self):
        self.parser.add_argument('--mimic_root', type=str)
        self.parser.set_defaults(mimic_root='/home/james/Documents/Charters/mimic-iii-clinical-database-1.4')
        self.parser.add_argument('--save_root', type=str)
        self.parser.set_defaults(save_root='/home/james/Documents/Charters/preprocessing_output')

    def parse(self):
        self.initialise()
        self.arguments = self.parser.parse_args()
        return self.arguments
