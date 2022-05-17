import json
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import torch
import pickle

def csv_to_dict(df: pd.DataFrame, key_col: str, val_col=str, two_ways: bool = True):
    key_val_dict = {}
    keys = list(df[key_col])
    vals = list(df[val_col])
    for k, v in zip(keys, vals):
        key_val_dict[k] = v

    if two_ways is True:
        val_key_dict = {}
        for k, v in zip(keys, vals):
            val_key_dict[v] = k
        return key_val_dict, val_key_dict
    else:
        return key_val_dict


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


def make_valid_path(name, is_dir=False, exist_ok=True):
    """
    This function make sure that a given path has all its parent directories created
    :param name: path name
    :param is_dir: True of this path is a directory and should be created also
    :param exist_ok: behaviour if the directory to be created is already existed
    :return: the same path passed to the function with its parents directories created
    """
    if is_dir is True:
        parent_dir = name
    else:
        parent_dir = os.path.dirname(name)
    os.makedirs(parent_dir, exist_ok=exist_ok)
    return name


def get_accelerator(device):
    """
    Get a torch device based on a string name of the target device
    :param device:
    :return:
    """
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def load_pickle(f_path):
    with open(f_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def dump_json(d: dict, outfile):
    with open(outfile, "w") as outfile:
        json.dump(d, outfile)

def dump_pickle(data, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump (data,f)

def standarize_features_set(fs):
    scaler = StandardScaler()
    fs = scaler.fit_transform(fs)
    return fs

def dict_to_str(d):
    str_dict = ''
    for k, v in d.items():
        str_dict = str_dict + f'{k}{v}--'
    return str_dict