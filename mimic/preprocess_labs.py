import os
import numpy as np
import pandas as pd
import pickle as pickle
import tqdm
from pprint import pprint
from sklearn.model_selection import train_test_split

from arguments import PreprocessingArguments


def preprocess_labs(args):
    print('*' * 17, 'preprocessor summoned for with the following settings:', sep='\n')
    pprint(vars(args), indent=2)

    # paths & dirs

    admissions_path = os.path.join(args.mimic_root, "ADMISSIONS.csv")
    labevents_path = os.path.join(args.mimic_root, "LABEVENTS.csv")
    d_labitems_path = os.path.join(args.mimic_root, "D_LABITEMS.csv")

    if not os.path.exists(args.save_root) or not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)

    # extract the labs only for non-na HADM_IDs

    labs = pd.read_csv(labevents_path)
    l = labs[labs.HADM_ID.isna() == False]
    l.loc[:, 'HADM_ID'] = l.HADM_ID.astype('int')
    l = l[l.VALUENUM.isna() == False]
    l.loc[:, 'VALUEUOM'] = l.VALUEUOM.astype('str')

    uom_scales = {
        50889: {"mg/L": 1, "mg/dL": 10, "MG/DL": 10},
        50916: {"ug/dL": 10, "nG/mL": 1},
        50926: {"mIU/L": 1, "mIU/mL": 1},
        50958: {"mIU/L": 1, "mIU/mL": 1},
        50989: {"pg/mL": 1, "ng/dL": 10},
        51127: {"#/uL": 1, "#/CU MM": 1},  # unclear #/CU MM RBC Ascites - dist looks roughly same.
        51128: {"#/uL": 1, "#/CU MM": 1},  # unclear #/CU MM WBC Ascites - dist looks roughly same.
    }

    def unitscale(itemid, valueuom):
        if (itemid in uom_scales) & (valueuom != 'nan'):
            scale_val_by = uom_scales[itemid][valueuom]
        else:
            scale_val_by = 1
        return scale_val_by

    l['SCALE'] = l.apply(lambda x: unitscale(x['ITEMID'], x['VALUEUOM']), axis=1)
    l['VALUE_SCALED'] = l['SCALE'] * l['VALUENUM']

    l.loc[:, ['HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE_SCALED']]\
        .to_csv("/home/james/Documents/Charters/labs/derived_labevents.csv")

    #lab_quantiles = l.head(10000000).groupby('ITEMID').VALUE_SCALED.quantile([0.1, 0.25, 0.75, 0.9])


if __name__ == "__main__":
    arguments = PreprocessingArguments().parse()
    preprocess_labs(arguments)
