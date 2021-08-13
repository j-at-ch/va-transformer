import os
import numpy as np
import pandas as pd
import pickle as pickle
import tqdm
from pprint import pprint
from sklearn.model_selection import train_test_split

from arguments import PreprocessingArguments


def preprocess(args):
    print('*' * 17, 'preprocessor summoned for with the following settings:', sep='\n')
    pprint(vars(args), indent=2)

    # paths

    chartevents_path = os.path.join(args.mimic_root, "CHARTEVENTS.csv")
    admissions_path = os.path.join(args.mimic_root, "ADMISSIONS.csv")
    d_items_path = os.path.join(args.mimic_root, "D_ITEMS.csv")

    if not os.path.exists(args.save_root) or not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)

    # read in admissions

    admissions = pd.read_csv(admissions_path,
                             index_col='ROW_ID',
                             parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'])

    # extract only those charted and apply labelling logic

    charted = admissions[admissions.HAS_CHARTEVENTS_DATA == 1]
    charted.drop('ROW_ID', axis=1, inplace=True)
    charted.loc[:, 'HADM_IN_SEQ'] = charted.groupby('SUBJECT_ID')['ADMITTIME'].rank().astype(int)
    charted = charted.sort_values(by=['SUBJECT_ID', 'HADM_IN_SEQ'])
    charted['ADMITTIME_NEXT'] = charted.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
    charted['DIS2ADM'] = charted['ADMITTIME_NEXT'] - charted['DISCHTIME']
    charted['READM<7'] = (charted['DIS2ADM'] < pd.Timedelta(days=7)).astype(int)
    charted['READM<30'] = (charted['DIS2ADM'] < pd.Timedelta(days=30)).astype(int)
    charted.set_index('HADM_ID', inplace=True)

    # get hadm_ids for the first admission

    first_indices = charted[charted.HADM_IN_SEQ == 1].index.to_numpy()

    # split first-hadm_ids into train, val, test and check.

    train_indices, surplus = train_test_split(first_indices, train_size=0.8)
    val_indices, test_indices = train_test_split(surplus, test_size=0.5)
    del surplus
    assert set(first_indices) == set(train_indices) | set(val_indices) | set(test_indices)

    # helpers

    def ts_to_posix(time):
        return pd.Timestamp(time, unit='s').timestamp()

    def get_admittime(hadm_id):
        time = charted.loc[hadm_id, 'ADMITTIME']
        return ts_to_posix(time)

    def get_from_charted(hadm_id, label):
        return charted.loc[hadm_id, label]

    # token mappings

    d_items = pd.read_csv(d_items_path)

    token_shift = 1
    pad_token = 0
    # eos_token = 1

    itemid2token = dict(zip(d_items['ITEMID'], range(token_shift, token_shift + len(d_items))))

    # add special tokens to the dictionary

    itemid2token['[PAD]'] = pad_token
    # itemid2token['[EOS]'] = 1
    # itemid2token['[BOS]'] = 1

    token2itemid = {v: k for k, v in itemid2token.items()}
    token2label = dict(zip(range(len(d_items)), d_items['LABEL']))

    def map2token(itemid):  # TODO: can now use data_utils.Mappings here.
        return itemid2token[np.int(itemid)]

    def map2itemid(token):
        return str(token2itemid[token])

    def map2itemidstr(tokens):
        return ' '.join(list(map(map2itemid, tokens)))

    # loop through sets and generate output files

    df = pd.read_csv(chartevents_path, skiprows=0, nrows=args.nrows,
                     header=0,
                     usecols=['HADM_ID', 'CHARTTIME', 'ITEMID'],
                     dtype={'HADM_ID': np.int},
                     converters={'ITEMID': map2token},
                     parse_dates=['CHARTTIME'])

    print('df read in!')

    for subset in ['val', 'train', 'test']:
        print(f'Processing {subset} set data...')

        # grouper for charts  # TODO: this is the bottleneck!

        gpdf = (df.query(f'HADM_ID.isin(@{subset}_indices)')
                .groupby(by='HADM_ID')
                )

        # count train token counts

        if subset == 'train':
            token2trcount = gpdf.obj.ITEMID.value_counts().to_dict()

        # initialise

        tokens = dict()
        times = dict()
        times_rel = dict()
        labels = dict()

        # populate with entries

        for i in tqdm.tqdm(gpdf.groups):
            time_origin = get_admittime(i)
            temp = gpdf.get_group(i).sort_values(by="CHARTTIME")
            tokens[i] = np.array(temp['ITEMID'], dtype=int)
            times[i] = np.fromiter(
                map(ts_to_posix, temp['CHARTTIME']),
                dtype=np.int64
            )
            times_rel[i] = times[i] - time_origin
            labels[i] = {
                'readm_7': get_from_charted(i, 'READM<7'),
                'readm_30': get_from_charted(i, 'READM<30')
            }

        # write out charts to pickle

        save_path = os.path.join(args.save_root, f'{subset}_charts.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_tokens': tokens,
                         f'{subset}_times': times,
                         f'{subset}_times_rel': times_rel}, f)
        del tokens, times, times_rel, gpdf

        # write out labels to pickle

        save_path = os.path.join(args.save_root, f'{subset}_labels.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_labels': labels}, f)
        del labels

        print(f'{subset} set data processed!')

    with open(os.path.join(args.save_root, 'mappings.pkl'), 'wb') as f:
        pickle.dump({'itemid2token': itemid2token,
                     'token2itemid': token2itemid,
                     'token2trcount': token2trcount},
                    f)


if __name__ == "__main__":
    arguments = PreprocessingArguments().parse()
    preprocess(arguments)
