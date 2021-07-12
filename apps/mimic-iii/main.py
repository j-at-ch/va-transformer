import torch
import methods
import data_utils
from arguments import Arguments
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    print('*'*17, 'chart-transformer called with the following settings:', sep='\n')
    pprint(vars(args), indent=2)


if __name__ == "__main__":
    arguments = Arguments().parse()
    main(arguments)
