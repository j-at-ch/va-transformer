# Preprocessing

## Required files:

+ Preprocessing mimic-iii-clinical-database-1.4 requires access to the MIMIC-III database. We will only need the following files
  + D_LABITEMS.csv
  + LABEVENTS.csv
  + ADMISSIONS.csv

To run preprocessing with the default arguments you will need to specify --mimic_root (which contains the files above) 
and --save_root (which is where the preprocessing output will be saved to). 

```
python preprocessing/mimic/preprocess_labs.py \
    --mimic_root <your/mimic/dir>\
    --save_root <your/desired/saved/root>\
```

## Some further details:
###Contract with training scripts

+ the 'main' scripts: pretraining.py, finetuning.py and baselining.py expect to find the following files in the directory\
specified by their `--data_root` argument. 

  + D_LABITEMS.csv
    + currently used by pretraining.py and finetuning.py for annotation of tokens
  + mappings.pkl
    + dict with keys 'itemid2token', 'token2itemid' 'token2trcount', each a dict mapping associated with tokens. 
  + train_data.pkl
    + dict with keys 'train_tokens', 'train_values', 'train_quants', 'train_times_rel' and each associated values is\
    a dict with keys given by (patient) index and values a numpy array. For each index, the arrays are temporally\
    aligned. 
  + train_targets.pkl
    + dict with key 'train_targets, value a dict which has keys index and values a dict which has keys \
    'DEATH>2.5D', 'DEATH>2.5D', 'DEATH<=3D', 'DEATH>3D', 'DEATH>7D', 'DEATH<=7D' and values either 0 or 1.
  + val_data.pkl
  + val_targets.pkl
  + test_data.pkl
  + test_targets.pkl
