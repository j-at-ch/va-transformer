import os

import numpy as np
import pandas as pd
import pickle as pickle
import tqdm

from pprint import pprint
from sklearn.model_selection import train_test_split
from preprocessing_arguments import PreprocessingArguments


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

    def convert_to_tokens(t, q):
        return tokqtok2ptokens[(t, q)]

#    X = map(convert_to_tokens, t1, q1)
#    X2 = np.fromiter(X, dtype=np.int32)


    mappings_save_path = os.path.join(args.save_root, "mappings_sc.pkl")

    with open(mappings_save_path, "wb") as f:
        pickle.dump({'itemid2token': tokqtok2ptokens,
                     'token2itemid': ptokens2tokqtok,
                     'token2trcount': None,
                     'qname2qtoken': None,
                     'qtoken2qname': None},
                    f)

"""
    from utils.data_utils import *
    tokens_train_np = fetch_data_as_numpy("/home/james/Documents/Charters/labs_dataset6/data/train_data.pkl",
                                          'train_tokens')
    quants_train_np = fetch_data_as_numpy("/home/james/Documents/Charters/labs_dataset6/data/train_data.pkl",
                                          'train_quants')

    # %%
    import numpy as np

    t1 = tokens_train_np[100001]
    q1 = quants_train_np[100001]

    x = np.concatenate(([t1], [q1]), axis=0)

    x[:, 0]

    # loop through index sets and generate output files
    for subset in ['train', 'val', 'test']:
        print(f'Processing {subset} set data...')

        # initialise
        tokens = dict()
        times = dict()
        times_rel = dict()
        values = dict()
        quants = dict()
        targets = dict()

        # populate with entries
        for i in tqdm.tqdm(groups.groups):
            temp = groups.get_group(i).sort_values(by="CHARTTIME")
            assert not temp.empty, f"Empty labs for hadm:{i}. There should be {get_from_adm(i, 'NUMLABS<2D')}"
            temp['QUANT'] = temp.apply(lambda x: apply_quantile_fct(x, lab_quantiles_train, 'VALUE_SCALED'), axis=1)

            tokens[i] = np.fromiter(
                map(map2token, temp['ITEMID']),
                dtype=np.int32
            )
            times[i] = np.fromiter(
                map(ts_to_posix, temp['CHARTTIME']),
                dtype=np.int64
            )

            admittime = get_from_adm(i, 'ADMITTIME')
            times_rel[i] = times[i] - ts_to_posix(admittime)

            values[i] = np.fromiter(
                temp['VALUE_SCALED'],
                dtype=np.float64
            )
            quants[i] = np.fromiter(
                temp['QUANT'],
                dtype=np.int32
            )

            # NOTE: can refactor target extraction easily to derive from augmented_admissions.csv
            los = pd.Timedelta(get_from_adm(i, 'LOS'))
            los = np.round(los.total_seconds()/(24*60*60) - 2, decimals=5)

            targets[i] = {
                'DEATH>2.5D': get_from_adm(i, 'DEATH>2.5D'),
                'DEATH<=3D': get_from_adm(i, 'DEATH<=3D'),
                'DEATH>3D': get_from_adm(i, 'DEATH>3D'),
                'DEATH>7D': get_from_adm(i, 'DEATH>7D'),
                'DEATH<=7D': get_from_adm(i, 'DEATH<=7D'),
                'LOS': los
            }

        # write out charts to pickle
        save_path = os.path.join(args.save_root, f'{subset}_data.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_tokens': tokens,
                         f'{subset}_values': values,
                         f'{subset}_quants': quants,
                         f'{subset}_times_rel': times_rel
                         },
                        f)
        del tokens, times, times_rel, groups

        # write out targets to pickle
        save_path = os.path.join(args.save_root, f'{subset}_targets.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_targets': targets}, f)
        del targets

        print(f'{subset} set data processed!')

    with open(os.path.join(args.save_root, 'mappings.pkl'), 'wb') as f:
        pickle.dump({'itemid2token': itemid2token,
                     'token2itemid': token2itemid,
                     'token2trcount': token2trcount,
                     'qname2qtoken': qname2qtoken,
                     'qtoken2qname': qtoken2qname},
                    f)

"""

if __name__ == "__main__":
    arguments = PreprocessingArguments().parse()
    postprocess(arguments)