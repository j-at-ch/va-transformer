import numpy as np
import pandas as pd
import pickle as pickle
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chartevents_path = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4/CHARTEVENTS.csv"
admissions_path = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4/ADMISSIONS.csv"
d_items_path = "C:/Users/james/Data/MIMIC/mimic-iii-clinical-database-1.4/d_items.csv"
save_path = 'C:/Users/james/Data/MIMIC/mimic-iii-chart-transformers/train_charts.pkl'

# read in admissions
admissions = pd.read_csv(admissions_path)
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])

# filter keep those that have chartevent data
admittimes = admissions[admissions.HAS_CHARTEVENTS_DATA == 1][['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]

# pick the earliest admission times for each subject.
admitfirsttimes = \
    pd.merge(admittimes[['SUBJECT_ID', 'ADMITTIME']] \
             .groupby(by='SUBJECT_ID').min(),
             admittimes,
             how='left',
             on=['SUBJECT_ID', 'ADMITTIME'])

assert len(admitfirsttimes) == admittimes['SUBJECT_ID'].nunique()
assert admitfirsttimes['HADM_ID'].nunique() == admitfirsttimes['SUBJECT_ID'].nunique()

# since we know HADM_ID is 1-1 with SUBJECT_ID, set as index
admitfirsttimes.set_index('HADM_ID', inplace=True)
all_indices = admitfirsttimes.index.to_numpy()

# split first-hadm_ids into train, val, test and check.

train_indices, surplus = np.split(all_indices, [int(0.8 * all_indices.size)])
val_indices, test_indices = np.split(surplus, [int(0.5 * surplus.size)])
del surplus
assert set(all_indices) == set(train_indices) | set(val_indices) | set(test_indices)


# helpers

def ts_to_posix(time):
    return pd.Timestamp(time, unit='s').timestamp()


def get_admittime(hadm_id):
    time = admitfirsttimes.loc[hadm_id, 'ADMITTIME']
    return ts_to_posix(time)


# grouper

gpdf = (pd.read_csv(chartevents_path, skiprows=0, nrows=100000,
                    header=0,
                    usecols=['HADM_ID', 'CHARTTIME', 'ITEMID'],
                    dtype={'HADM_ID': np.int, 'ITEMID': str},
                    parse_dates=['CHARTTIME'])
        .query('HADM_ID.isin(@train_indices)')
        .groupby(by='HADM_ID')
        )

# initialise

train_items = dict()
train_times = dict()
train_times_rel = dict()

# populate with entries

for i in gpdf.groups:
    time_origin = get_admittime(i)
    temp = gpdf.get_group(i).sort_values(by="CHARTTIME")
    train_items[i] = np.array(temp['ITEMID'], dtype=int)
    train_times[i] = np.fromiter(
        map(ts_to_posix, temp['CHARTTIME']),
        dtype=np.int64
    )
    train_times_rel[i] = train_times[i] - time_origin

# note - need to be sure of whole sequence before normalising.

# write out dicts to pickle
with open(save_path, 'wb') as f:
    pickle.dump(
        {'train_items': train_items,
         'train_times': train_times,
         'train_times_rel': train_times_rel
         },
        f
    )