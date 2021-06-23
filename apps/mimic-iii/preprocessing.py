
import os
import numpy as np
import pandas as pd
import pickle as pickle
import torch
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chartevents_path = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4/CHARTEVENTS.csv"
admissions_path = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4/ADMISSIONS.csv"
d_items_path = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4/d_items.csv"
save_root = 'C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers'


# read in admissions
admissions = pd.read_csv(admissions_path)
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])

# filter keep those that have chartevent data
admittimes = admissions[admissions.HAS_CHARTEVENTS_DATA == 1][['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]

# pick the earliest admission times for each subject.
admitfirsttimes = pd.merge(admittimes[['SUBJECT_ID', 'ADMITTIME']].groupby(by='SUBJECT_ID').min(),
                           admittimes,
                           how='left',
                           on=['SUBJECT_ID', 'ADMITTIME'])

assert len(admitfirsttimes) == admittimes['SUBJECT_ID'].nunique()
assert admitfirsttimes['HADM_ID'].nunique() == admitfirsttimes['SUBJECT_ID'].nunique()

# TODO: insert logic extracting target labels

# since we know HADM_ID is 1-1 with SUBJECT_ID, set as index
admitfirsttimes.set_index('HADM_ID', inplace=True)
all_indices = admitfirsttimes.index.to_numpy()

# split first-hadm_ids into train, val, test and check.

train_indices, surplus = train_test_split(all_indices, train_size=0.8)
val_indices, test_indices = train_test_split(surplus, test_size=0.5)
del surplus
assert set(all_indices) == set(train_indices) | set(val_indices) | set(test_indices)

# helpers


def ts_to_posix(time):
    return pd.Timestamp(time, unit='s').timestamp()


def get_admittime(hadm_id):
    time = admitfirsttimes.loc[hadm_id, 'ADMITTIME']
    return ts_to_posix(time)


# token mappings

d_items = pd.read_csv(d_items_path)

token_shift = 10
pad_token = 0

itemid2token = dict(zip(d_items['ITEMID'], range(token_shift, token_shift + len(d_items))))

# add special tokens to the dictionary
itemid2token['[PAD]'] = 0

token2itemid = {v: k for k, v in itemid2token.items()}
token2label = dict(zip(range(len(d_items)), d_items['LABEL']))


with open(os.path.join(save_root, 'mappings.pkl'), 'wb') as f:
    pickle.dump({'itemid2token': itemid2token,
                 'token2itemid': token2itemid},
                f)

def map2token(itemid):
    return itemid2token[np.int(itemid)]


def map2itemid(token):
    return str(token2itemid[token])


def map2itemidstr(tokens):
    return ' '.join(list(map(map2itemid, tokens)))


# loop through sets and generate output files

for subset in ['val', 'train', 'test']:
    print(f'Processing {subset} set data...')

    # grouper

    gpdf = (pd.read_csv(chartevents_path, skiprows=0, nrows=1000000,
                        header=0,
                        usecols=['HADM_ID', 'CHARTTIME', 'ITEMID'],
                        dtype={'HADM_ID': np.int},
                        converters={'ITEMID': map2token},
                        parse_dates=['CHARTTIME'])
            .query(f'HADM_ID.isin(@{subset}_indices)')
            .groupby(by='HADM_ID')
            )

    # initialise

    tokens = dict()
    times = dict()
    times_rel = dict()

    # populate with entries

    for i in gpdf.groups:
        time_origin = get_admittime(i)
        temp = gpdf.get_group(i).sort_values(by="CHARTTIME")
        tokens[i] = np.array(temp['ITEMID'], dtype=int)
        times[i] = np.fromiter(
            map(ts_to_posix, temp['CHARTTIME']),
            dtype=np.int64
        )
        times_rel[i] = times[i] - time_origin

    # write out dicts to pickle

    save_path = os.path.join(save_root, f'{subset}_charts.pkl')

    with open(save_path, 'wb') as f:
        pickle.dump({f'{subset}_tokens': tokens,
                     f'{subset}_times': times,
                     f'{subset}_times_rel': times_rel}, f)

    del tokens, times, times_rel, gpdf

    print(f'{subset} set data processed!')
