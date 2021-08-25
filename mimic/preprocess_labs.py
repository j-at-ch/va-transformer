import os
import numpy as np
import pandas as pd
import pickle as pickle
import tqdm

from sklearn.model_selection import train_test_split

from arguments import PreprocessingArguments


def preprocess_labs(args):
    # paths

    admissions_path = os.path.join(args.mimic_root, "ADMISSIONS.csv")
    labevents_path = os.path.join(args.mimic_root, "LABEVENTS.csv")
    d_labitems_path = os.path.join(args.mimic_root, "D_LABITEMS.csv")
    targets_path = os.path.join(args.save_root, "augmented_admissions.csv")

    # read admissions

    admissions = (pd.read_csv(admissions_path,
                              index_col='HADM_ID',
                              parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'])
                  .drop(['ROW_ID'], axis=1)
                  .rename(columns={'HAS_CHARTEVENTS_DATA': 'HAS_CHARTS'})
                  )

    # read labevents and summarise

    labevents = (pd.read_csv(labevents_path,
                             index_col='ROW_ID',
                             parse_dates=['CHARTTIME'])
                 .dropna(subset=['HADM_ID'])
                 .astype({'HADM_ID': 'int', 'VALUEUOM': 'str'})
                 )

    labsinfo = pd.DataFrame(
        {'NUMLABS': labevents.groupby('HADM_ID').ITEMID.count(),
         'NUMLABVALS': labevents.groupby('HADM_ID').VALUENUM.count(),
         'FIRSTLABTIME': labevents.groupby('HADM_ID').CHARTTIME.min(),
         'LASTLABTIME': labevents.groupby('HADM_ID').CHARTTIME.max()}
    )

    adm = pd.concat([admissions, labsinfo], axis=1)
    adm.loc[:, 'HAS_LABS'] = (~adm.NUMLABS.isna()).astype('int')
    adm.loc[:, 'HADM_IN_SEQ'] = adm.groupby('SUBJECT_ID')['ADMITTIME'].rank().astype(int)
    adm.loc[:, 'LOS'] = (adm.DISCHTIME - adm.ADMITTIME)
    adm.loc[:, 'ADMIT_TO_EXPIRE'] = (adm.DEATHTIME - adm.ADMITTIME)
    adm.loc[:, 'EXPIRE_BEFORE_ADMIT'] = (adm.ADMIT_TO_EXPIRE < pd.Timedelta(days=0)).astype('int')
    adm.loc[:, 'DEATH>1D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=1)).astype('int')
    adm.loc[:, 'DEATH>2.5D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=2.5)).astype('int')
    adm.loc[:, 'DEATH<=3D'] = (adm.ADMIT_TO_EXPIRE <= pd.Timedelta(days=3)).astype('int')
    adm.loc[:, 'DEATH>3D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=3)).astype('int')
    adm.loc[:, 'DEATH<=7D'] = (adm.ADMIT_TO_EXPIRE <= pd.Timedelta(days=7)).astype('int')
    adm.loc[:, 'DEATH>7D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=7)).astype('int')
    adm.loc[:, 'DEATH>10D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=10)).astype('int')

    # add in calculations relying on a join between ADMISSIONS summaries and LABEVENTS:

    labevents = labevents.join(adm[['ADMITTIME']], on='HADM_ID')
    labevents_2d = labevents[labevents.CHARTTIME <= labevents.ADMITTIME + pd.Timedelta(days=2)]

    adm = pd.concat([adm,
                     pd.DataFrame({
                         'NUMLABS<2D': labevents_2d.groupby('HADM_ID').ITEMID.count(),
                         'NUMLABVALS<2D': labevents_2d.groupby('HADM_ID').VALUENUM.count()
                     })
                     ], axis=1
                    )

    print(f"writing augmented admissions df to {targets_path}...")
    adm.to_csv(targets_path)
    print("written!\n")

    # select hadms for this data slice:

    hadms = adm[(
            (adm.LOS >= pd.Timedelta(days=2)) & (adm['NUMLABS<2D'] > 0)
            & (pd.isna(adm.ADMIT_TO_EXPIRE) | (adm.ADMIT_TO_EXPIRE >= pd.Timedelta(days=2)))
            # remove deaths occurring during stay and before 2 days.
    )].index.to_numpy()

    # split first-hadm_ids into train, val, test and assert that they partition.

    train_indices, surplus = train_test_split(hadms, train_size=0.8, random_state=1965)
    val_indices, test_indices = train_test_split(surplus, test_size=0.5, random_state=1965)
    del surplus
    assert set(hadms) == set(train_indices) | set(val_indices) | set(test_indices)
    print(f"num_train: {len(train_indices)}, num_val: {len(val_indices)}, num_test: {len(test_indices)}")

    # ready the tokens:

    d_labitems = pd.read_csv(d_labitems_path)

    special_tokens = {'[PAD]': 0}
    token_shift = len(special_tokens)
    itemid2token = dict(zip(d_labitems['ITEMID'], range(token_shift, token_shift + len(d_labitems))))
    itemid2token.update(special_tokens)

    token2itemid = {v: k for k, v in itemid2token.items()}

    def map2token(itemid):
        return itemid2token[int(itemid)]

    def ts_to_posix(time):
        return pd.Timestamp(time, unit='s').timestamp()

    def get_from_adm(hadm_id, target):
        return adm.loc[hadm_id, target]

    # process the labs

    uom_scales = {
        50889: {"mg/L": 1, "mg/dL": 10, "MG/DL": 10},
        50916: {"ug/dL": 10, "nG/mL": 1},
        50926: {"mIU/L": 1, "mIU/mL": 1},
        50958: {"mIU/L": 1, "mIU/mL": 1},
        50989: {"pg/mL": 1, "ng/dL": 10},
        51127: {"#/uL": 1, "#/CU MM": 1},  # unclear #/CU MM RBC Ascites - distr looks roughly same.
        51128: {"#/uL": 1, "#/CU MM": 1},  # unclear #/CU MM WBC Ascites - distr looks roughly same.
    }

    def unitscale(itemid, valueuom):  # TODO: induces major bottleneck - solve.
        if (itemid in uom_scales) & (valueuom != 'nan'):
            scale_val_by = uom_scales[itemid][valueuom]
        else:
            scale_val_by = 1
        return scale_val_by

    def get_numeric_quantile_from_(quantiles_df, itemid, value):
        if itemid not in quantiles_df.index:
            return -1
        q = quantiles_df.loc[itemid]
        array = (value <= q)
        if value > q.iloc[-1]:
            index = len(q)
        elif not any(array):
            index = -1
        else:
            a, = np.where(array)
            index = a[0]
        return index

    def apply_quantile_fct(df, quantiles_df):
        if pd.isna(df.VALUENUM):
            return -1
        else:
            return get_numeric_quantile_from_(quantiles_df, df.ITEMID, df.VALUENUM)

    print("unit-scaling lab values...")
    labevents['SCALE'] = labevents.apply(lambda x: unitscale(x['ITEMID'], x['VALUEUOM']), axis=1)
    labevents['VALUE_SCALED'] = labevents['SCALE'] * labevents['VALUENUM']
    print("lab values unit-scaled!\n")

    # loop through index sets and generate output files

    for subset in ['train', 'val', 'test']:
        print(f'Processing {subset} set data...')

        # grouper for charts

        groups = (labevents.query(f'HADM_ID.isin(@{subset}_indices)')
                  .groupby(by='HADM_ID')
                  )

        # train token counts

        if subset == 'train':
            print("counting train token frequencies...")
            token2trcount = (groups.obj['ITEMID']
                             .apply(map2token)
                             .value_counts()
                             .to_dict()
                             )
            print("train token frequencies counted!\n")

            print("calculating train lab value quantiles...")
            lab_quantiles_train = groups.obj.groupby('ITEMID').VALUE_SCALED.quantile([0.1, 0.25, 0.75, 0.9])
            print("train lab value quantiles calculated!\n")

        # initialise

        tokens = dict()
        times = dict()
        times_rel = dict()
        values = dict()
        quantiles = dict()
        targets = dict()

        # populate with entries

        for i in tqdm.tqdm(groups.groups):
            admittime = get_from_adm(i, 'ADMITTIME')
            temp = groups.get_group(i).sort_values(by="CHARTTIME")
            temp = temp[temp.CHARTTIME <= admittime + pd.Timedelta(days=2)]
            assert not temp.empty, f"Empty labs for hadm:{i}. There should be {get_from_adm(i, 'NUMLABS<2D')}"
            temp['QUANTILE'] = temp.apply(lambda x: apply_quantile_fct(x, lab_quantiles_train), axis=1)

            tokens[i] = np.fromiter(
                map(map2token, temp['ITEMID']),
                dtype=np.int32
            )
            times[i] = np.fromiter(
                map(ts_to_posix, temp['CHARTTIME']),
                dtype=np.int64
            )
            times_rel[i] = times[i] - ts_to_posix(admittime)
            values[i] = temp['VALUENUM']
            quantiles[i] = temp['QUANTILE']

            # NOTE: can refactor target extraction easily to derive from augmented_admissions.csv
            targets[i] = {
                'DEATH>2.5D': get_from_adm(i, 'DEATH>2.5D'),
                'DEATH<=3D': get_from_adm(i, 'DEATH<=3D'),
                'DEATH>3D': get_from_adm(i, 'DEATH>3D'),
                'DEATH>7D': get_from_adm(i, 'DEATH>7D'),
                'DEATH<=7D': get_from_adm(i, 'DEATH<=7D'),
                'LOS': get_from_adm(i, 'LOS')
            }

        # write out charts to pickle

        save_path = os.path.join(args.save_root, f'{subset}_data.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_tokens': tokens,
                         f'{subset}_values': values,
                         f'{subset}_quantiles': quantiles,
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
                     'token2trcount': token2trcount},
                    f)


if __name__ == "__main__":
    arguments = PreprocessingArguments().parse()
    preprocess_labs(arguments)
