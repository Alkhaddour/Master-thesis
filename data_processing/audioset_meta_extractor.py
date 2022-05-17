import sys

from utils.data_utils import train_test_split
sys.path.append('..')

import glob
import os
import pandas as pd
import torchaudio
from tqdm import tqdm

from utils.display_utils import info


def is_valid(filepath):
    try:
        waveform, _ = torchaudio.load(filepath)
        waveform = waveform[0, :]  # take single channel
    except:
        info(f"Failed to load file @ {filepath}")
    return True


def get_existed_files(data_dir, validate_content=True):
    invalid=0
    existed_files = []
    info("Validating files")
    for filepath in tqdm(glob.glob(f'{data_dir}/*/*.wav')):
        relpath = os.path.relpath(filepath, start=data_dir)
        if validate_content == True and not is_valid(filepath):
            invalid +=1
            continue
        existed_files.append(relpath)
    if validate_content:
        print(f"Number of invalid files = {invalid}")

    return existed_files


def get_meta_data(data_dir):
    """
    Create meta file from all csv files
    :param data_dir: Directory to Audioset audio files and csv annotations
    :param out_path: full path to save the resulting csv meta file
    """
    def get_class_from_filename(filename):
        return filename.split("/")[0]
    paths = []
    classes = []
    for file in tqdm(os.listdir(data_dir)):
        if not file.endswith('.csv'):
            continue
        class_dir = file[:-4]
        csv_file = pd.read_csv(os.path.join(data_dir, file), header=None, names=['prefix', 'st', 'et', 'classes'])
        prefixes = list(csv_file['prefix'])
        sts = list(csv_file['st'])
        batch_paths = [os.path.join(class_dir, f"{p}_{st}.wav") for (p, st) in zip(prefixes, sts)]
        paths = paths + batch_paths
        classes = classes + list(csv_file['classes'])
    df = pd.DataFrame(list(zip(paths, classes)),
                      columns=['audio_path', 'classes']).drop_duplicates()
    df = df.loc[df['audio_path'].isin(get_existed_files(data_dir))]
    df['label'] = df['audio_path'].apply(lambda row: get_class_from_filename(row))
    return df


def sample_meta(df,  max_per_class=100):
    available_labels= list(df['label'].unique())
    selected_samples = []
    for label in available_labels:
        tmp_df = df.loc[df['label'] == label]                                       # get class instances
        selected_samples.append(tmp_df.sample(min(max_per_class, len(tmp_df)),      # sample at most max_per_class
                                              random_state=1))    
    return pd.concat(selected_samples)                                              # concatenate and return the extracted sample
    

if __name__ == '__main__':
    max_per_class=100 # max samples to choose per class
    info("Indexing existing files")
    md = get_meta_data('../data/AudioSet/raw')  # './AudioSet/audioset_metafile.csv'
    md = sample_meta(md, max_per_class=max_per_class)
    info(f"Sampled {len(md)} classes")
    info("Splitting into train/test")
    train_data, test_data = train_test_split(md)
    info(f"Samples in train data {len(train_data)}")
    info(f"Samples in test data {len(test_data)}")
    md.to_csv(f'../data/AudioSet/audioset_meta_{max_per_class}.csv', index=False)
    train_data.to_csv(f'../data/AudioSet/audioset_meta_{max_per_class}_train.csv', index=False)
    test_data.to_csv(f'../data/AudioSet/audioset_meta_{max_per_class}_test.csv',  index=False)



