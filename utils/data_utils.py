import pandas as pd
from sklearn import model_selection


def extract_meta_csv(meta_dict):
    """
    meta_dict is a dict with 3 keys:
    'files': list of N files
    'chunk_cnt': integer C_i represents number of chunks in audio N_i
    'clustering_labels': list of M labels where M = sum C_i
    """
    assert len(meta_dict['files']) == len(meta_dict['chunk_cnt']), "You didn't expect that ;) did u?"
    files = []
    chucks = []
    for i in range(len(meta_dict['files'])):
        files = files + [meta_dict['files'][i]] * meta_dict['chunk_cnt'][i]
        chucks = chucks + list(range(0, meta_dict['chunk_cnt'][i]))

    meta_csv = pd.DataFrame(list(zip(files,chucks, meta_dict['clustering_labels'])),
                            columns =['files','chucks', 'clustering_labels'])
    return meta_csv

def train_test_split(meta_data: pd.DataFrame, label_col='label', train_pct=0.75):
    train_data, test_data, _, _ = model_selection.train_test_split(meta_data, meta_data[label_col], train_size=train_pct)
    return train_data, test_data