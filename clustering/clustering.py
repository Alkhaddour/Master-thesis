
import os, sys
sys.path.append('..')

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from configs.global_config import PROJECT_DIR
import pickle
from utils.basic_utils import make_valid_path, standarize_features_set
from utils.display_utils import info, blank_line


def kmeans_clustering(features, n_clusters=128):
    """
    Clusterig 23598 embedding vector using K-MEANS takes 30sec
    """
    return KMeans(n_clusters=n_clusters, random_state=0).fit(features).labels_

def spectral_clustering(features, n_clusters=128):
    """
    [2022-03-28 16:45:07.558844] -- Clusterig 23598 embedding vector using SPECTRAL
    UserWarning: Graph is not fully connected, spectral embedding may not work as expected.
    [2022-03-28 16:57:08.790600] -- Done!
    """
    return SpectralClustering(n_clusters=n_clusters,
                              assign_labels='discretize', 
                              affinity='nearest_neighbors',
                              n_jobs=-1,
                              verbose=True
                              ).fit(features).labels_

def dbscan_clustering(features, min_samples=30):
    """
    eps is very important param which should be tuned carefully
    """
    return DBSCAN(eps=0.001, min_samples=min_samples).fit(features).labels_

pca_files = ['PCA_512_to_32',
               'PCA_512_to_64',
               'PCA_512_to_128',
               'PCA_6144_to_32',
               'PCA_6144_to_64',
               'PCA_6144_to_128'
               ]
ALGO = 'K-MEANS' # choose one of those: 'K-MEANS', 'SPECTRAL', 'DBSCAN'

for file in pca_files:
    for N_CLUSTERS in [64, 128]:
        # Trace run
        info(f"Running {ALGO} clustering on file {file} with n_clusters = {N_CLUSTERS}")
        # input params
        in_data_path = os.path.join(PROJECT_DIR, f'data/AudioSet/splits_PCA/{file}.pkl')
        # output files
        clustering_res_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/splits_clusters/')
        make_valid_path(clustering_res_dir, is_dir=True)
        splits_labels_path = os.path.join(clustering_res_dir, f'F--{file}_C--{N_CLUSTERS}_A--{ALGO}.pkl') # features, clusters and algorithm

        info("Load features from disk")
        with open(in_data_path, 'rb') as f:
            pca_X = pickle.load(f)

        info("Standarization")    
        features = standarize_features_set(pca_X['features'])

        info(f"Clusterig {len(features)} embedding vector using {ALGO}")
        if ALGO == 'K-MEANS':
            lbls = kmeans_clustering(features, n_clusters=N_CLUSTERS)
        elif ALGO == 'SPECTRAL':
            lbls = spectral_clustering(features, n_clusters=N_CLUSTERS)
        elif ALGO == 'DBSCAN':
            lbls = dbscan_clustering(features, n_clusters=N_CLUSTERS)
        else:
            raise ValueError(f"Unexpected clustering algorithm, expected one of ['K-MEANS', 'SPECTRAL', 'DBSCAN'] got {ALGO}.")

        res= {'files': pca_X['files'] , 'chunk_cnt':pca_X['chunk_cnt'], 'clustering_labels': lbls}
        with open(splits_labels_path, 'wb') as f:
            pickle.dump(res, f)
        blank_line()

info("Done!")



