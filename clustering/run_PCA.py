import os, sys

from sklearn.decomposition import PCA
sys.path.append('../')
from data_processing.datasets import load_features

from utils.basic_utils import make_valid_path

from utils.display_utils import info
import pickle

if __name__ == "__main__":
    for src_features_size in [512, 6144]:
        for N_COMPONENTS in [32, 64, 128]:
            info(f"Running PCA from {src_features_size} to {N_COMPONENTS}")
            # inputs
            features_meta_path = '../data/AudioSet/splits_meta.csv'
            splits_root_dir = f'../data/AudioSet/splits/{src_features_size}'
            # outputs
            pca_root_dir = make_valid_path(f'../data/AudioSet/splits_PCA/', is_dir=True)
            pca_out_file = os.path.join(pca_root_dir, f'PCA_{src_features_size}_to_{N_COMPONENTS}.pkl')
            # data holder
            chunks = []

            # load features
            files, chunk_cnt, features = load_features(splits_root_dir, features_meta_path)
            info(f"Features size: {features.size()}")
            info(f"files count: {len(files)}")

            # predict PCA and save to disk
            pca = PCA(n_components=N_COMPONENTS, svd_solver='full')
            info(f"Applying PCA...")
            info(f"Old features dims : {features.shape}")
            new_features = pca.fit_transform(features)
            info(f"New features dims : {new_features.shape}")

            info("Saving to disk")
            pca_X = {'files' :files,
                    'chunk_cnt': chunk_cnt,
                    'features': new_features,
                    }

            with open(pca_out_file, 'wb') as f:
                pickle.dump(pca_X, f)

    info("Done")






