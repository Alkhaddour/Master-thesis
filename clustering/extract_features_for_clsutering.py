import sys
sys.path.append('..')


import os
import numpy as np
import openl3
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from utils.basic_utils import make_valid_path
from utils.display_utils import info


def generate_features_outfile(infile, ext='npy'):
    outfile = infile[:-3] + ext
    return outfile


if __name__ == "__main__":
    # inputs
    audioset_meta_path = '../data/AudioSet/audioset_meta_100.csv'
    audioset_root_dir = '../data/AudioSet/raw'
    features_size = 512

    # outputs
    features_root_dir = make_valid_path(f'../data/AudioSet/splits/{features_size}', is_dir=True)
    features_meta_path = '../data/AudioSet/splits_meta.csv'
    error_file_path = os.path.join('../data/AudioSet', f'errors_{features_size}.csv')
    audio_name = []
    n_splits = []
    audio_classes = []
    failed_to_process = {'filename':[], 'error': []}
    # load metadata
    audioset_meta = pd.read_csv(audioset_meta_path)

    # load on files, extract featurs, append meta to list, save features to disk
    for idx in tqdm(range(len(audioset_meta))):
        row = audioset_meta.iloc[idx]
        infile = row['audio_path']
        inpath = os.path.join(audioset_root_dir, infile)
        try:
            # This is the reader used in openl3 tutorial
            audio, sr = sf.read(inpath)
            # set hop_size=1 to extract features from non-intersecting segments
            emb, _ = openl3.get_audio_embedding(audio, sr, embedding_size=features_size, content_type="env", center=True, hop_size=1.0, verbose=False)
        except Exception as e:
            info(f"Error processing file @ {infile}")
            failed_to_process['filename'].append(infile)
            failed_to_process['error'].append(str(e))
            continue

        outfile = generate_features_outfile(infile)
        # update meta
        audio_name.append(outfile)
        n_splits.append(emb.shape[0])
        audio_classes.append(row['classes'])
        # save features
        outpath = make_valid_path(os.path.join(features_root_dir, outfile))
        np.save(outpath, emb)

    # save meta file to disk
    df = pd.DataFrame(list(zip(audio_name, n_splits, audio_classes)), 
                      columns =['audio_name', 'n_splits', 'classes'])
    df.to_csv(features_meta_path, index=False)
    pd.DataFrame.from_dict(failed_to_process).to_csv(error_file_path, index=False)
    
