import pickle
import torch


def fetch_mappings(mapping_path):
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    return mappings


def fetch_data_as_torch(path, var_key):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    di = data[var_key]
    return {k: torch.from_numpy(v) for k, v in di.items()}
