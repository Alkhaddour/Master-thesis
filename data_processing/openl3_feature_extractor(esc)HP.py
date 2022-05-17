"""
In this file we will extract features from AudioSet files. This script will process all audioset files and save the resulting embeddings in one pickle file.

OUTPUT:

TYPE: dict 
VALUE:    {'CLASS_X/FILE_NAME_1': [LIST, OF, VALUES],
            'CLASS_X/FILE_NAME_2': [LIST, OF, VALUES],
            ...
            'CLASS_Y/FILE_NAME_1': [LIST, OF, VALUES],
          }
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from utils.basic_utils import dump_pickle, load_pickle, make_valid_path

from configs.global_config import PROJECT_DIR
import openl3
import librosa
from tqdm import tqdm
import tensorflow as tf
def extract_openl3_emebeddings(audio, sr, embedding_size,hop_size=1.0):
     emb, ts = openl3.get_audio_embedding(audio, sr, embedding_size=embedding_size, content_type="env", center=True, hop_size=hop_size, verbose=False)
     return emb, ts

def assert_length(waveform, target_len=1, sr = 16000, type='repeat'):
    wav_len = len(waveform)
    tar_len_samples =  target_len * sr
    if wav_len >= tar_len_samples:
        return waveform
    append_len = tar_len_samples - wav_len

    if type=='repeat':
        return np.append(waveform, waveform[:append_len])
    elif type =='zeros':
        return np.append(waveform, np.zeros(append_len))
    else:
        raise Exception(f"unimplemented append of type {type}")


# inputs
esc_raw_dir = os.path.join(PROJECT_DIR, 'data/ESC-50/raw/no_silence/audio_16k')
embeddings = {'sr': 16000, 'embedding_size':512 ,'hop_size':0.2, 'content_type': 'env'}
min_wav_len = 16000

esc10_H_files = list(pd.read_csv('/home/alhasan/workdir/SED/data/ESC-50/esc10_H.csv')['filename'])
esc10_P_files = list(pd.read_csv('/home/alhasan/workdir/SED/data/ESC-50/esc10_P.csv')['filename'])


#outputs
esc_openl3_embeddings_dir = os.path.join(PROJECT_DIR, f"data/ESC-50/esc10(nosilence)_openl3_{int(embeddings['hop_size']*1000)}_{embeddings['embedding_size']}_HP_regular/")
make_valid_path(esc_openl3_embeddings_dir)

skipped = 0
with tf.device('/GPU:1'):
    # outputs
    progress_bar = tqdm([dir for dir in os.listdir(esc_raw_dir) if dir.endswith('.wav')])
    for filename in progress_bar:
        audio_path = os.path.join(esc_raw_dir, filename)        
        (waveform, sr) = librosa.core.load(audio_path, sr=embeddings['sr'], mono=True)
        y_harmonic, y_percussive = librosa.effects.hpss(waveform)
        if filename in esc10_H_files:
            waveform = y_harmonic
        elif filename in esc10_P_files:
            waveform = y_percussive

        else:
            skipped +=1
            print(filename)
            continue
        
        waveform = assert_length(waveform, target_len=1, sr = sr, type='zeros')
        embs,ts = extract_openl3_emebeddings(waveform, 
                                        sr=embeddings['sr'], 
                                        embedding_size = embeddings['embedding_size'],
                                        hop_size = embeddings['hop_size']
                                        )
        embs = embs[:-1, :]
        progress_bar.set_description(f"{filename} {type(embs)} of size: {embs.shape}")
        np.save(os.path.join(esc_openl3_embeddings_dir, filename), embs)
    progress_bar.set_description("")

print("skipped: ", skipped)