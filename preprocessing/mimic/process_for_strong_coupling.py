import os

import numpy as np
import pickle as pickle
import tqdm

from pprint import pprint
from preprocessing_arguments import PreprocessingArguments


def fetch_data_as_numpy(path, var_key):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    di = data[var_key]
    return di


def postprocess(args):
    print('*' * 17, 'processing data for strong-coupling experiment with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # define and save mapping dicts

    mappings_path = os.path.join(args.data_root, "mappings.pkl")

    with open(mappings_path, "rb") as f:
        mappings_dict = pickle.load(f)

    del mappings_dict['itemid2token']['[PAD]']
    del mappings_dict['token2itemid'][0]
    del mappings_dict['qname2qtoken']['[PAD]']
    del mappings_dict['qtoken2qname'][0]

    l1 = len(mappings_dict['itemid2token'])
    l2 = len(mappings_dict['qname2qtoken'])

    itemval2ptokens = dict(zip(
        [(i, j)
         for i in mappings_dict['itemid2token']
         for j in mappings_dict['qname2qtoken']],
        range(1, l1 * l2)
    ))
    itemval2ptokens[('[PAD]', '[PAD]')] = 0
    ptokens2itemval = {v:k for k, v in itemval2ptokens.items()}

    tokqtok2ptokens = dict(zip(
        [(i, j)
         for i in mappings_dict['token2itemid']
         for j in mappings_dict['qtoken2qname']],
        range(1, l1 * l2)
    ))
    tokqtok2ptokens[(0, 0)] = 0
    ptokens2tokqtok = {v: k for k, v in tokqtok2ptokens.items()}

    def convert_pair_to_tokens(t, q):
        return tokqtok2ptokens[(t, q)]

    ptokens2count = {k: 0 for k in ptokens2tokqtok}

    # loop through index sets and generate output files
    for subset in ['train', 'val', 'test']:
        print(f'Processing {subset} set data...')

        # data

        data_path = os.path.join(args.data_root, f'{subset}_data.pkl')
        tokens_np = fetch_data_as_numpy(data_path, f'{subset}_tokens')
        quants_np = fetch_data_as_numpy(data_path, f'{subset}_quants')
        times_rel = fetch_data_as_numpy(data_path, f'{subset}_times_rel')

        # initialise
        
        tokens_sc = dict()

        # populate with entries

        for i in tqdm.tqdm(tokens_np):
            tokens_sc[i] = np.fromiter(
                map(convert_pair_to_tokens, tokens_np[i], quants_np[i]),
                dtype=np.int32
            )

            if subset == 'train':
                for tok in tokens_sc[i]:
                    ptokens2count[tok] += 1

        # reverse-sort ptokens2count

        ptokens2count = dict(sorted(ptokens2count.items(), key=lambda item: item[1], reverse=True))
        ptokens2count = {k:v for k, v in ptokens2count.items() if v > 0}
        print(ptokens2count)

        save_path = os.path.join(args.save_root, f'{subset}_data_sc.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_tokens': tokens_sc,
                         f'{subset}_times_rel': times_rel
                         },
                        f)
        del tokens_np, tokens_sc, quants_np, times_rel

        print(f'{subset} set data processed!')

    mappings_save_path = os.path.join(args.save_root, "mappings_sc.pkl")
    with open(mappings_save_path, "wb") as f:
        pickle.dump({'itemid2token': tokqtok2ptokens,
                     'token2itemid': ptokens2tokqtok,
                     'token2trcount': ptokens2count,
                     'qname2qtoken': None,
                     'qtoken2qname': None},
                    f)


if __name__ == "__main__":
    arguments = PreprocessingArguments().parse()
    postprocess(arguments)
