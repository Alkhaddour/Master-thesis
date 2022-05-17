
import os

from torch import load
from configs.global_config import PROJECT_DIR
from data_processing.datasets import ClusteringDataset
from torch.utils.data import DataLoader
from configs.data_configs import get_audioset_config_v1 as data_config
from models.performer import PerformerSED
from utils.basic_utils import load_pickle
from dotted_dict import DottedDict
import pandas as pd
import numpy as np

from utils.data_utils import extract_meta_csv
if __name__ == "__main__":
    x = np.load('/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512/1-110389-A-0.wav.npy')
    print(x)
    # labels = load_pickle('/home/alhasan/workdir/SED/data/AudioSet/PANN_labels.pkl')
    # used_classes = []
    # print(type(labels['labels']))
    # for _,v in labels['labels'].items():
    #     used_classes = used_classes + list(v)
    # used_classes_set = list(set(used_classes))
    # labels_file = pd.read_csv('/home/alhasan/workdir/SED/data/AudioSet/class_labels_indices.csv')
    # class_names = list(labels_file['display_name'])

    # occ = {class_names[i]: used_classes.count(i) for i in used_classes_set}
    # occ= {k: v for k, v in sorted(occ.items(), key=lambda item: item[1])}
    # print(occ)

    # meta_dict = load_pickle('/home/alhasan/workdir/SED/data/AudioSet/splits_clusters/F--PCA_512_to_32_C--128_A--K-MEANS.pkl')
   
    # print(len(meta_dict['files']))  
    # print(len(meta_dict['chunk_cnt']))
    # print("-----------")
    # # convert list of tuples to list
    # # fs = []    
    # # for t in meta_dict['files']:
    # #     fs = fs + list(t)
    # # meta_dict['files'] = fs
    # print(len(meta_dict['files']))
    # print(len(meta_dict['chunk_cnt']))
    # # transform meta dict
    # meta_csv = extract_meta_csv(meta_dict)
    # meta_csv.to_csv("Saved.csv", index=False)
    # meta_csv = meta_csv.loc[meta_csv['clustering_labels'] != 70]#.to_csv('sth.csv', index=False)

    # available_clusters = list(meta_csv['clustering_labels'].unique())
    # meta_csv['clustering_labels'] = meta_csv['clustering_labels'].apply(available_clusters.index)
    # print(max(meta_csv['clustering_labels']))
    # # --------------------------
    # meta_csv.loc[meta_csv['clustering_labels'] == 1].to_csv('sth.csv', index=False)
    # meta_csv.groupby('clustering_labels')['files'].count().reset_index().rename(columns={'files':'count'}).sort_values('count').to_csv('tmp.csv', index =False)
    
    #     # Exp path to evaluate
    # exp_dir = '../outputs/Audioset--clustering_pretrain--V1.4/'
    # # load config from file
    # load_config_file = True
    # if load_config_file:
    #     configs = load_pickle(os.path.join(exp_dir, 'configs.pkl'))
    #     tr_data_cfg = DottedDict(configs['tr_data_cfg'])
    #     te_data_cfg = DottedDict(configs['te_data_cfg'])
    #     train_cfg = DottedDict(configs['train_cfg'])
    #     model_cfg = DottedDict(configs['model_cfg'])
    # else:
    #     raise Exception("Unimplemented file")
    
    # root_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/raw')
    # train_ds= ClusteringDataset(root_dir=root_dir, 
    #                         meta_file=tr_data_cfg.meta_path,
    #                         data_cfg=tr_data_cfg,
    #                         mode='train')
    # test_ds= ClusteringDataset(root_dir=root_dir, 
    #                             meta_file=te_data_cfg.meta_path,
    #                             data_cfg=te_data_cfg,
    #                             mode='test')

    # train_loader = DataLoader(train_ds, batch_size=train_cfg.val_batch_size, shuffle=False, num_workers=train_cfg.num_workers,
    #                           pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=train_cfg.val_batch_size, shuffle=False, num_workers=train_cfg.num_workers,
    #                          pin_memory=True)
    # # define model
    # audio_model = PerformerSED(dim=64, 
    #                            depth=1, 
    #                            heads=1, 
    #                            out=model_cfg.out)
    # print(audio_model)