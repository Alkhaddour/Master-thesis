"""
In this model we should write code for pretraining all combinations we already suggested in Description.txt
"""
from datetime import datetime
from torch.utils.data import DataLoader
from torch import optim

import torch.nn as nn
import sys
import os

sys.path.append('../../')
from data_processing.datasets import ClusteringDataset
from utils.basic_utils import dump_json, dump_pickle, make_valid_path
from configs.global_config import OUTPUT_PATH, PROJECT_DIR
from models.transformer import TransformerSED
from utils.managers import ModelManager, TrainManager
from configs.data_configs import get_clustering_config_v1 as data_config
from configs.train_configs import get_config_v1 as train_config
from configs.model_configs import get_transformer_config_v1 as model_config
from utils.display_utils import info

def run_experiment(lr, n_class, dim, n_heads, depth, mlp_dim, dim_head, dropout, emb_dropout, train_csv_file, test_csv_file, device):
    exp_prefix = f"exp_1/dim{dim}_heads{n_heads}_depth{depth}_mlpdim{mlp_dim}_lr{lr}_({os.path.basename(train_csv_file).split('__')[0]})/"
    # load configs
    tr_data_cfg = data_config({'meta_path': train_csv_file,
                                'timem': 3200,  # maximum possible length of the mask. (200ms),
                                'vsize': 'all',
                                'freqm': 24,  # maximum possible length of the mask.
                                'num_mel_bins': 128,
                                'target_length':128,
                                'n_class': n_class
                                })

    te_data_cfg = data_config({'meta_path': test_csv_file,
                                'timem': 0,  # maximum possible length of the mask. (200ms),
                                'vsize': 'all',
                                'freqm': 0,  # maximum possible length of the mask.
                                'num_mel_bins': 128,
                                'target_length':128,
                                'n_class': n_class
                                })
    assert dim == tr_data_cfg.target_length, "Hey bro, it is the input size, are u OK?"
    # Define datasets
    root_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/raw')
    train_ds = ClusteringDataset(root_dir=root_dir, 
                                meta_file=tr_data_cfg.meta_path,
                                data_cfg=tr_data_cfg,
                                mode='train')
    test_ds = ClusteringDataset(root_dir=root_dir, 
                                meta_file=te_data_cfg.meta_path,
                                data_cfg=te_data_cfg,
                                mode='test')


    train_cfg = train_config({'n_epochs': 100, 
                                'patience': 30, 
                                'n_print_steps': 50,  
                                'batch_size': 128, 
                                'val_batch_size' :128, 
                                'device': device, 
                                'lr':lr})

    model_cfg = model_config({'dim':dim, 
                                'depth':depth, 
                                'heads':n_heads, 
                                'mlp_dim': mlp_dim,
                                'dim_head': dim_head,
                                'out':tr_data_cfg.n_class,
                                'dropout': dropout,
                                'emb_dropout': emb_dropout})

    # define model
    audio_model = TransformerSED(n_mels=tr_data_cfg.num_mel_bins, 
                                    num_classes=model_cfg.out, 
                                    dim=model_cfg.dim, 
                                    depth=model_cfg.depth, 
                                    heads=model_cfg.heads,  
                                    mlp_dim=model_cfg.mlp_dim, 
                                    dim_head=model_cfg.dim_head,
                                    dropout = model_cfg.dropout,
                                    emb_dropout = model_cfg.emb_dropout
                                )
    # optimizers
    optimizer = optim.Adam(audio_model.parameters(), lr=train_cfg.lr)
    loss_func = nn.CrossEntropyLoss()
    scheduler = None

    # loaders
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.val_batch_size, shuffle=True, num_workers=train_cfg.num_workers,
                            pin_memory=True)

    # define output directories
    exp_dir = os.path.join(OUTPUT_PATH, exp_prefix)
    model_name = 'pretrained'
    make_valid_path(exp_dir, is_dir=True, exist_ok=True)

    # save configs
    configs = {'tr_data_cfg': tr_data_cfg,
                'te_data_cfg': te_data_cfg,
                'train_cfg': train_cfg,
                'model_cfg': model_cfg}
    json_out = os.path.join(exp_dir, 'configs.json')
    pkl_out = os.path.join(exp_dir, 'configs.pkl')

    info(f"Saving JSON configs to {json_out}")
    dump_json(configs, outfile=json_out)
    dump_pickle(configs, outfile=pkl_out)


    # define Managers
    model_mgr = ModelManager(model_name, exp_dir, progress_log_train=train_cfg.n_print_steps,
                            progress_log_val=train_cfg.n_print_steps, patience=train_cfg.patience)

    train_mgr = TrainManager(n_epochs=train_cfg.n_epochs,
                            loss_fn=loss_func,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            device=train_cfg.device,
                            model_manager=model_mgr,
                            verbose=False)
    info(f">>>>>>>>> Training cfg: {exp_prefix}")
    train_mgr.train_model(audio_model, train_loader=train_loader, val_loader=test_loader)
    return str(exp_prefix), train_mgr.model_manager.best_val_loss

def get_ds_meta(openl3_emb, pca_dim, n_clusters, mode): 
    return f'/home/alhasan/workdir/SED/data/AudioSet/splits_clusters/dev_meta/F--PCA_{openl3_emb}_to_{pca_dim}_C--{n_clusters}_A--K-MEANS__{mode}.csv'

def clear_run_file(runID):
    with open(os.path.join(OUTPUT_PATH, f'exp_1/summary_{runID}.txt'), 'w') as f:
        f.write('')
    with open(os.path.join(OUTPUT_PATH, f'exp_1/log_{runID}.txt'), 'w') as f:
        f.write('')
if __name__ == '__main__':
    make_valid_path(os.path.join(OUTPUT_PATH, 'exp_1'), is_dir=True, exist_ok=True)
    runID = f'run{sys.argv[1]}'
    clear_run_file(runID)
    runs= {'run0': ['cuda:0', 1e-5, 128],
           'run1': ['cuda:0', 1e-5, 64],
           'run2': ['cuda:1', 1e-4, 128],
           'run3': ['cuda:1', 1e-4, 64]
          }
    device = runs[runID][0]
    lr = runs[runID][1]
    n_clusters = runs[runID][2]
    for depth in [4, 3, 2]:
        for heads in [8, 4, 2]:
            for mlp_dim in [1024, 512, 256]:
                for openl3_emb in [6144, 512]:
                    for pca_dim in [128, 64, 32]:
                            try:
                                test_name, loss = run_experiment(lr=lr, n_class= n_clusters, dim=128, n_heads=heads, depth=depth, mlp_dim=mlp_dim, dim_head=64, dropout=0.1, emb_dropout=0.0, 
                                                    train_csv_file = get_ds_meta(openl3_emb, pca_dim, n_clusters, mode='train'), 
                                                    test_csv_file = get_ds_meta(openl3_emb, pca_dim, n_clusters, mode='val'), 
                                                    device=device)
                                with open(os.path.join(OUTPUT_PATH, f'exp_1/summary_{runID}.txt'), 'a') as f:
                                    f.write(f'{test_name}, {loss}\n')
                            except Exception as e:
                                print(e)
                                with open(os.path.join(OUTPUT_PATH, f'exp_1/log_{runID}.txt'), 'a') as f:
                                    f.write(f'[{datetime.now()}] -- {e}\n\n')

                            
    
