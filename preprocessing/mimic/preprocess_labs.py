import os
import sys

import numpy as np
import pandas as pd
import pickle as pickle
import tqdm

from pprint import pprint
from sklearn.model_selection import train_test_split
from preprocessing_arguments import PreprocessingArguments


def ts_to_posix(time):
    return pd.Timestamp(time, unit='s').timestamp()


uom_scales = {
        50889: {"mg/L": 1, "mg/dL": 10, "MG/DL": 10},
        50916: {"ug/dL": 10, "nG/mL": 1},
        50926: {"mIU/L": 1, "mIU/mL": 1},
        50958: {"mIU/L": 1, "mIU/mL": 1},
        50989: {"pg/mL": 1, "ng/dL": 10},
        51127: {"#/uL": 1, "#/CU MM": 1},
        51128: {"#/uL": 1, "#/CU MM": 1},
    }


def unitscale(itemid, valueuom):  # TODO: implement more efficient solution.
    if (itemid in uom_scales) & (valueuom != 'nan'):
        scale_val_by = uom_scales[itemid][valueuom]
    else:
        scale_val_by = 1
    return scale_val_by


def get_numeric_quantile_from_(quantiles_df, itemid, value):
    # maps unknown indices to 0
    # otherwise maps to 1..(num_quantiles+1)
    if itemid not in quantiles_df.index:
        index = -1
    else:
        q = quantiles_df.loc[itemid]  # q is a quants series
        array = (value <= q)
        if value > q.iloc[-1]:
            index = len(q)
        else:
            a, = np.where(array)
            index = a[0]
    return index + 1


def apply_quantile_fct(labs_df, quantiles_df):
    if pd.isna(labs_df.VALUENUM):
        return 0
    else:
        return get_numeric_quantile_from_(quantiles_df, labs_df.ITEMID, labs_df.VALUENUM)


def preprocess_labs(args):
    print('*' * 17, 'preprocessing labs with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    labevents_path = os.path.join(args.mimic_root, "LABEVENTS.csv")
    d_labitems_path = os.path.join(args.mimic_root, "D_LABITEMS.csv")

    # read labevents and summarise

    labevents = (pd.read_csv(labevents_path,
                             index_col='ROW_ID',
                             parse_dates=['CHARTTIME'])
                 .dropna(subset=['HADM_ID'])
                 .astype({'HADM_ID': 'int', 'VALUEUOM': 'str'})
                 )

    if args.augmented_admissions == 'w':
        admissions_path = os.path.join(args.mimic_root, "ADMISSIONS.csv")
        targets_path = os.path.join(args.save_root, "augmented_admissions.csv")

        # read admissions

        admissions = (pd.read_csv(admissions_path,
                                  index_col='HADM_ID',
                                  parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'])
                      .drop(['ROW_ID'], axis=1)
                      .rename(columns={'HAS_CHARTEVENTS_DATA': 'HAS_CHARTS'})
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
                             'NUMLABVALS<2D': labevents_2d.groupby('HADM_ID').VALUENUM.count(),
                             'NUMLABSUNQ<2D': labevents_2d.groupby('HADM_ID').ITEMID.nunique()
                         })
                         ], axis=1
                        )

        # select hadms for this data slice:

        qualifying_hadms = adm[(
            (adm.LOS >= pd.Timedelta(days=2)) & (adm['NUMLABS<2D'] >= args.min_num_labs)
            & (pd.isna(adm.ADMIT_TO_EXPIRE) | (adm.ADMIT_TO_EXPIRE >= pd.Timedelta(days=2)))
        )]
        qualifying_hadm_ids = qualifying_hadms.index.to_numpy()

        qualifying_subjects = qualifying_hadms.loc[:, 'SUBJECT_ID'].unique()
        print(qualifying_subjects, len(qualifying_subjects))

        # split qualifying subject_ids into train, val, test and assert that they partition.

        train_subjects, surplus = train_test_split(qualifying_subjects, train_size=0.8, random_state=1965)
        val_subjects, test_subjects = train_test_split(surplus, train_size=0.5, random_state=1965)
        del surplus

        train_indices = qualifying_hadms[qualifying_hadms.SUBJECT_ID.isin(train_subjects)].index.to_numpy()
        val_indices = qualifying_hadms[qualifying_hadms.SUBJECT_ID.isin(val_subjects)].index.to_numpy()
        test_indices = qualifying_hadms[qualifying_hadms.SUBJECT_ID.isin(test_subjects)].index.to_numpy()

        print(f"num_train: {len(train_indices)}",
              f"num_val: {len(val_indices)}",
              f"num_test: {len(test_indices)}",
              f"num_qualifying: {len(qualifying_hadm_ids)}")
        assert set(qualifying_hadm_ids) == set(train_indices) | set(val_indices) | set(test_indices)

        adm.loc[train_indices, 'PARTITION'] = 'train'
        adm.loc[val_indices, 'PARTITION'] = 'val'
        adm.loc[test_indices, 'PARTITION'] = 'test'

        print(f"writing augmented admissions df to {targets_path}...")
        adm.to_csv(targets_path)
        print("written!\n")

    else:  # dev: what do I need if relying on aug_adm?
        admissions_path = os.path.join(args.data_root, "augmented_admissions.csv")
        adm = pd.read_csv(admissions_path,
                          index_col='HADM_ID',
                          parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'])

    labevents = labevents.join(adm[['ADMITTIME']], on='HADM_ID')
    labevents_2d = labevents[labevents.CHARTTIME <= labevents.ADMITTIME + pd.Timedelta(days=2)]

    train_indices = adm[adm.PARTITION == 'train'].index.to_numpy()
    val_indices = adm[adm.PARTITION == 'val'].index.to_numpy()
    test_indices = adm[adm.PARTITION == 'test'].index.to_numpy()

    # ready the tokens:

    d_labitems = pd.read_csv(d_labitems_path)

    special_tokens = {'[PAD]': 0}
    token_shift = len(special_tokens)
    itemid2token = dict(zip(d_labitems['ITEMID'], range(token_shift, token_shift + len(d_labitems))))
    itemid2token.update(special_tokens)

    token2itemid = {v: k for k, v in itemid2token.items()}

    def map2token(itemid):
        return itemid2token[int(itemid)]

    def get_from_adm(hadm_id, target):
        return adm.loc[hadm_id, target]

    # minor u.o.m. processing needed for particular labs

    if args.scale_labs:
        print("unit-scaling lab values...")
        labevents['SCALE'] = labevents.apply(lambda x: unitscale(x['ITEMID'], x['VALUEUOM']), axis=1)
        labevents['VALUE_SCALED'] = labevents['SCALE'] * labevents['VALUENUM']
        print("lab values unit-scaled!\n")
        labs_out_path = os.path.join(args.save_root, "labevents_scaled.csv")
        print(f"writing scaled labs df to {labs_out_path} for posterity...")
        labevents.to_csv(labs_out_path)
        print("written!\n")
    else:
        print("lab values are not being rescaled!")

    if args.generating_data_for == "1.5D":
        # loop through index sets and generate output files
        for subset in ['train', 'val', 'test']:
            print(f'Processing {subset} set data...')

            # grouper for labs
            groups = (labevents.query(f'HADM_ID.isin(@{subset}_indices)')
                      .groupby(by='HADM_ID')
                      )

            # train token counts and quantile calculation
            if subset == 'train':
                print("counting train token frequencies...")
                token2trcount = (groups.obj['ITEMID']
                                 .apply(map2token)
                                 .value_counts()
                                 .to_dict()
                                 )
                print("train token frequencies counted!\n")

                print("calculating train lab value quants...")
                lab_quantiles_train = groups.obj.groupby('ITEMID').VALUE_SCALED.quantile([0.1, 0.25, 0.75, 0.9])
                print("train lab value quants calculated!\n")

            # initialise
            tokens = dict()
            times = dict()
            times_rel = dict()
            values = dict()
            quants = dict()
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
                values[i] = np.fromiter(
                    temp['VALUENUM'],
                    dtype=np.float64
                )
                quants[i] = np.fromiter(
                    temp['QUANTILE'],
                    dtype=np.int32
                )

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
                             f'{subset}_quantiles': quants,
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


def preprocess_labs_for_1D(args):
    print('*' * 17, 'preprocessing labs with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    admissions_path = os.path.join(args.data_root, "augmented_admissions.csv")
    scaled_labevents_path = os.path.join(args.data_root, "scaled_LABEVENTS.csv")
    lab_quantiles_path = os.path.join(args.save_root, "lab_quantiles.csv")
    d_labitems_path = os.path.join(args.mimic_root, "D_LABITEMS.csv")

    # read labevents and summarise

    labevents = (pd.read_csv(scaled_labevents_path,
                             index_col='ROW_ID',
                             parse_dates=['CHARTTIME', 'ADMITTIME'])
                 .dropna(subset=['HADM_ID'])
                 .astype({'HADM_ID': 'int', 'VALUEUOM': 'str'})
                 )

    adm = pd.read_csv(admissions_path,
                      index_col='HADM_ID',
                      parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'])

    train_indices = adm[adm.PARTITION == 'train'].index.to_numpy()
    val_indices = adm[adm.PARTITION == 'val'].index.to_numpy()
    test_indices = adm[adm.PARTITION == 'test'].index.to_numpy()

    with open(os.path.join(args.data_root, 'mappings.pkl'), 'rb') as f:
        itemid2token = pickle.load(f)['itemid2token']

    def map2token(itemid):
        return itemid2token[int(itemid)]

    def get_from_adm(hadm_id, target):
        return adm.loc[hadm_id, target]

    labevents_2d = labevents[labevents.CHARTTIME <= labevents.ADMITTIME + pd.Timedelta(days=2)]

    for subset in ['train', 'val', 'test']:
        print(f'Processing {subset} set data...')

        # grouper for labs
        groups = (labevents_2d.query(f'HADM_ID.isin(@{subset}_indices)')
                  .groupby(by='HADM_ID')
                  )

        # train token counts and quantile calculation
        if (subset == 'train') & bool(args.write_quantiles_summary):
            print("calculating train lab value quants...")
            lab_quantiles_train = groups.obj.groupby('ITEMID').VALUE_SCALED.quantile(args.quantiles)
            print("train lab value quants calculated!\n")
            lab_quantiles_train.to_csv(lab_quantiles_path)
            print(f"train lab quantiles info written to {lab_quantiles_path}\n")
        elif not bool(args.write_quantiles_summary):
            lab_quantiles_train = pd.read_csv(lab_quantiles_path)

        # initialise
        values_mean = dict()
        values_latest = dict()
        values_count = dict()
        quants = dict()

        print(labevents_2d)

        # populate with entries
        for i in tqdm.tqdm(groups.groups):
            temp = groups.get_group(i).sort_values(by="CHARTTIME")

            print(temp.groupby('ITEMID')['VALUE_SCALED'].count())
            print(temp.groupby('ITEMID')['VALUE_SCALED'].tail(1))
            print(temp.groupby('ITEMID')['VALUE_SCALED'].mean())
            print(temp[temp.ITEMID == '51519'])

            assert 0 == 1

            #assert not temp.empty, f"Empty labs for hadm:{i}. There should be {get_from_adm(i, 'NUMLABS<2D')}"
            #temp['QUANT'] = temp.apply(lambda x: apply_quantile_fct(x, lab_quantiles_train), axis=1)
            """
            tokens[i] = np.fromiter(
                map(map2token, temp['ITEMID']),
                dtype=np.int32
            )
            times[i] = np.fromiter(
                map(ts_to_posix, temp['CHARTTIME']),
                dtype=np.int64
            )
            times_rel[i] = times[i] - ts_to_posix(admittime)
            values[i] = np.fromiter(
                temp['VALUENUM'],
                dtype=np.float64
            )
            quants[i] = np.fromiter(
                temp['QUANTILE'],
                dtype=np.int32
            )
            """
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


class Data1p5D:  # dev
    def __init__(self, tokens, values=None, quants=None, times=None):
        self.tokens = tokens  # dict with k:v as hadm: np.array
        self.values = values
        self.quants = quants
        self.times = times


class Targets:  # dev
    def __init__(self, target, **kwargs):
        self.target = target


class Data1D:  # dev
    def __init__(self, values, quants=None, **kwargs):
        self.values = values  # dict with k:v as hadm: np.array
        self.quants = quants


if __name__ == "__main__":
    arguments = PreprocessingArguments().parse()
    #preprocess_labs(arguments)
    preprocess_labs_for_1D(arguments)
