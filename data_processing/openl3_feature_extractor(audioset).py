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


sys.path.append('../')
from utils.basic_utils import dump_pickle, load_pickle

from configs.global_config import PROJECT_DIR
import openl3
import librosa
from tqdm import tqdm
def extract_openl3_emebeddings(audio, sr, embedding_size,hop_size=1.0):
     emb, ts = openl3.get_audio_embedding(audio, sr, embedding_size=embedding_size, content_type="env", center=True, hop_size=hop_size, verbose=False)
     return emb, ts

def extend_waveform(waveform, length=15680, type='repeat'):
    if type=='repeat':
        return np.append(waveform, waveform[-length:])
    else:
        raise Exception(f"unimplemented append of type {type}")


# inputs
as_raw_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/raw')

for es in [512]:
    # outputs
    embeddings = {'extraction_config' : {'sr': 16000, 'embedding_size':es ,'hop_size':0.5, 'content_type': 'env'},
                'embeddings': {}}


    progress_bar = tqdm([dir for dir in os.listdir(as_raw_dir) if dir.endswith('.csv') == False])
    for dir in progress_bar:
        if dir.endswith('.csv'):
            continue
        audio_dir = os.path.join(as_raw_dir, dir)
        for file in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, file)
            short_name = os.path.join(dir, file)
            progress_bar.set_description(short_name)
            
            (waveform, _) = librosa.core.load(audio_path, sr=embeddings['extraction_config']['sr'], mono=True)
            # waveform = extend_waveform(waveform, length=7680, type='repeat') 
            embs,ts = extract_openl3_emebeddings(waveform, 
                                            sr=embeddings['extraction_config']['sr'], 
                                            embedding_size = embeddings['extraction_config']['embedding_size'],
                                            hop_size = embeddings['extraction_config']['hop_size']
                                            )
            embs = embs[:-1, :]
            embeddings['embeddings'][short_name] = embs

        progress_bar.set_description("")

    openl3_embeddings_path = os.path.join(PROJECT_DIR, f"data/AudioSet/openl3_embeddings_{int(embeddings['extraction_config']['hop_size']*1000)}_{embeddings['extraction_config']['embedding_size']}.pkl")
    dump_pickle(embeddings, openl3_embeddings_path)