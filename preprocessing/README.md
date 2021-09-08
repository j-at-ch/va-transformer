# Preprocessing

## Contract with training scripts

+ the 'main' scripts: pretraining.py, finetuning.py and baselining.py expect to find the following files in the directory\
specified by their `--data_root` argument. 

  + D_LABITEMS.csv
    + currently used by pretraining.py and finetuning.py for annotation of tokens
  + mappings.pkl
    + dict with keys 'itemid2token', 'token2itemid' 'token2trcount', each a dict mapping associated with tokens. 
  + train_data.pkl
    + dict with keys 'train_tokens', 'train_values', 'train_quantiles', 'train_times_rel' and each associated values is\
    a dict with keys given by (patient) index and values a numpy array. For each index, the arrays are temporally\
    aligned. 
  + train_targets.pkl
    + dict with key 'train_targets, value a dict which has keys index and values a dict which has keys \
    'DEATH>2.5D', 'DEATH>2.5D', 'DEATH<=3D', 'DEATH>3D', 'DEATH>7D', 'DEATH<=7D' and values either 0 or 1.
  + val_data.pkl
  + val_targets.pkl
  + test_data.pkl
  + test_targets.pkl

### TODOs
+ The items above would be better as classes. Then the contract between the output of the preprocessing would be\ 
cleaner: we would just need to check that the preprocess_* scripts end by pickling instances of the classes, and that
the model expects the classes. rf. current method is to pickle into dicts and then read the dicts into classes.\ 
cf. `data_utils.Mappings`