"""
In this file we will extract labels from AudioSet files. This script will process all audioset files and save the resulting labels in one pickle file.

OUTPUT:

TYPE: dict 
VALUE:    {'extraction_config': {'CFG_1': 'VAL_1', 'CFG_2': 'VAL_2', ...},
           'labels': {'CLASS_X/FILE_NAME_1': [LIST, OF, VALUES],
                      'CLASS_X/FILE_NAME_2': [LIST, OF, VALUES],
                      ...
                      'CLASS_Y/FILE_NAME_1': [LIST, OF, VALUES],
                     }
          }
"""
import sys

sys.path.append('../')
import os
from configs.global_config import PANNS_PATH, PROJECT_DIR
from utils.basic_utils import dump_pickle, load_pickle
from models.PANNs import PANNs_labeler
from tqdm import tqdm
import pandas as pd
# inputs
as_raw_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/raw')

# outputs
pann_labels_path = os.path.join(PROJECT_DIR, 'data/AudioSet/PANN_labels_all(new).pkl')
panns_labels = {'extraction_config' : None,
                'labels': None}

# define PANNS model
panns_config = {'sr':16000, 'ws':1024,'hs':320, 'mb':64, 'fmn':50, 'fmx':14000}
labeler = PANNs_labeler(sr=panns_config['sr'], ws=panns_config['ws'],hs=panns_config['hs'], mb=panns_config['mb'], 
                        fmn=panns_config['fmn'], fmx=panns_config['fmx'], model_chkpt=PANNS_PATH, device='cuda:0', 
                        labeling_accuracy=1.0, labeling_hop=0.5, min_coverage=0.7, labels=None, verbose=False)

labels = {}
meta = {} # dict of class and count

progress_bar = tqdm([dir for dir in os.listdir(as_raw_dir) if dir.endswith('.csv') == False])
for dir in progress_bar:
    if dir.endswith('.csv'):
        continue
    audio_dir = os.path.join(as_raw_dir, dir)
    for file in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, file)
        short_name = os.path.join(dir, file)
        progress_bar.set_description(short_name)
        hard_output, h_labels = labeler.extract_label(audio_path=audio_path)
        labels[short_name] = hard_output

        for out in hard_output:
            if out in meta:
                meta[out] = meta[out] + 1
            else:
                meta[out] = 1
    progress_bar.set_description("")

panns_labels = {'extraction_config' : panns_config,
                'labels': labels}

dump_pickle(panns_labels, pann_labels_path)
meta = {k: v for k, v in sorted(meta.items(), key=lambda item: item[1])}


# print classes distribution to file
labels_map = pd.read_csv('/home/alhasan/workdir/SED/data/AudioSet/class_labels_indices.csv')
nones = 0
with open('/home/alhasan/workdir/SED/data/AudioSet/pann_labels_all_count(new).csv', 'w') as f:
    for class_, count in meta.items():
        if class_ != None:
            f.write(f"{labels_map.iloc[class_]['display_name']:30s} > {str(class_):10s} > {count}\n")
        else:
            nones += 1
    f.write(f"Number of ignored parts (None parts) = {nones}\n")


