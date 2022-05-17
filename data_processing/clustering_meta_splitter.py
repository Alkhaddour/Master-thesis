import os, sys

sys.path.append('..')
from utils.display_utils import info
from configs.global_config import PROJECT_DIR
from utils.basic_utils import load_pickle, make_valid_path
from utils.data_utils import extract_meta_csv, train_test_split
# read pickle file,  convert to expanded csv, split it to train val, save files

pkls_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/splits_clusters/')
for pkl_file in os.listdir(pkls_dir):
    if pkl_file.endswith('.pkl') == False:
        continue
    else:
        info(f"Processing file: {pkl_file}")

    # inputs
    clustering_pkl= os.path.join(pkls_dir, pkl_file)
    delete_clusters = [] # if you want to delete any clsuters

    # outputs
    dev_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/splits_clusters/dev_meta')
    train_fname = f'{pkl_file[:-4]}__train.csv'  # csv 
    train_path = os.path.join(dev_dir, train_fname)
    val_fname = f'{pkl_file[:-4]}__val.csv'      # csv
    val_path = os.path.join(dev_dir, val_fname)

    # init
    dev_dir = make_valid_path(dev_dir, is_dir=True)
    # load pkl
    meta_data = extract_meta_csv(load_pickle(clustering_pkl))

    # preprocess
    if isinstance(delete_clusters, list):
        meta_data = meta_data.loc[~meta_data['clustering_labels'].isin(delete_clusters)]
    # remap lables in case we deleted some labels
    available_clusters = list(meta_data['clustering_labels'].unique())
    meta_data['clustering_labels'] = meta_data['clustering_labels'].apply(available_clusters.index)

    # split data 
    train_data, val_data = train_test_split(meta_data, label_col='clustering_labels')

    # save to dir
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
