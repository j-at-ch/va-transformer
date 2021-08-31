# chart-transformers

Motivation is patient chart event-based data.

Reproducibility guarantee: the authors are confident that the results of this code are reproducible.

## Features
Using this repo you can:
+ pretrain a cutting edge decoder-style transformer-based model
+ finetune the pretrained model for a specific classification task

and be able to interactively visualise:
+ the development of token embeddings and training logs with tensorboard

## How to use this repo

### Set-up

Users might wish to set some of the command-line options as defaults in arguments.py.
Having set up your venv from the requirements file, 

### Preprocessing

```
python mimic/processing.py \
    --mimic_root \
    --save_root \
    --nrows \
```

### Pretraining commands

The core arguments for pretraining are:
```
python mimic/pretraining.py \
    --data_root <path_to_preprocessed_data_dir>\
    --save_root <path_to_desired_save_dir>\
    --logs_root <path_to_desired_log_dir>\
    --attn_depth 6\
    --attn_dim 100\
    --attn_heads 8\
    --seq_len 200\
    --model_name 'pretraining_test'\
    --num_epochs 50
```

Provided you specify the correct `--data_root` and give a valid `--save_root` and `--logs_root`, the model should run
straight out of the box.

*HINT*: to do a test run, append the option `--test_run True` to the command above.

To see all specifiable options, type:
```
python mimic/pretraining.py --help 
```
and for further details consult `mimic/arguments.py` which shows the default settings.

### Finetuning

The core arguments for finetuning are:
```
python mimic/finetuning.py \
    --data_root <path_to_preprocessed_data_dir>\
    --save_root <path_to_desired_save_dir>\
    --logs_root <path_to_desired_log_dir>\
    --attn_depth 6\
    --attn_dim 100\
    --attn_heads 8\
    --seq_len 200\
    --model_name 'finetuning_test'\
    --num_epochs 50
    --pretuned_model <path_to_ckpt_of_output_of_pretraining>
```

*HINT*: to do a test run, append the option `--test_run True` to the command above.


### Visualisation

Tensorboard. Make sure that you do not have tensorflow in your venv. This will confuse tensorboard.
If you wish to see the embeddings, make sure to append model_name to logs, as below.
```
tensorboard --logdir <path_to_desired_log_dir>/model_name
```

## Requirements

We have provided a requirements.txt specifying the basic packages needed to use this repo. We use a conda env, 
but have also tested with pip. Note that Anaconda does not provide a distribution for the entmax package. 
This is accessible on pypi. To bring it into a conda-managed venv
you can add pip to your conda venv. Then install entmax into venv directly via pip. E.g.

`conda create --name <chart-env>`

`conda activate <chart-env>`

`conda install --file requirements.txt`

`anaconda3/envs/<chart-env>/bin/pip install entmax`


### Models

vg1
vg1.1
vg1.2
vg1.3
vg1.4 only difference from vg1.3 is --quantile_pad_token set to 5 so that it is separate from others. 