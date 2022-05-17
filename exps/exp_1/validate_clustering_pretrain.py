"""
Validate best model we got in our experiment
"""
import os 
import sys

sys.path.append('../../')
from models.transformer import TransformerSED

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing.datasets import ClusteringDataset
from utils.managers import ModelManager
from utils.display_utils import info
from utils.model_utils import Metrics, get_best_model_from_path
from dotted_dict import DottedDict
from utils.basic_utils import load_pickle
from configs.global_config import OUTPUT_PATH, PROJECT_DIR


def validate_clustering_pretrained_model(model, data_loader, device, thrs=None):
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    for step, (data, target) in tqdm(enumerate(data_loader)):
        # Find logits
        output = model(data.to(device)) 
        # find argmax
        output = torch.argmax(output, dim=1)     
        # append to list
        y_true.append(target.detach().cpu())
        y_pred.append(output.detach().cpu())

    avg_metrics = Metrics.calculate_multi_class_metrics(torch.hstack(y_true),torch.hstack(y_pred), n_classes=64)                                                                                
    return avg_metrics

if __name__ == "__main__":
    # Exp path to evaluate
    exp_dir = os.path.join(OUTPUT_PATH, 'exp_1/dim128_heads8_depth3_mlpdim1024_lr0.0001_(F--PCA_512_to_32_C--64_A--K-MEANS)')
    # load config from file
    load_config_file = True
    if load_config_file:
        configs = load_pickle(os.path.join(exp_dir, 'configs.pkl'))
        tr_data_cfg = DottedDict(configs['tr_data_cfg'])
        te_data_cfg = DottedDict(configs['te_data_cfg'])
        train_cfg = DottedDict(configs['train_cfg'])
        model_cfg = DottedDict(configs['model_cfg'])
    else:
        raise Exception("Unimplemented file")
    
    root_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/raw')
    train_ds= ClusteringDataset(root_dir=root_dir, 
                            meta_file=tr_data_cfg.meta_path,
                            data_cfg=tr_data_cfg,
                            mode='train')
    test_ds= ClusteringDataset(root_dir=root_dir, 
                                meta_file=te_data_cfg.meta_path,
                                data_cfg=te_data_cfg,
                                mode='test')

    train_loader = DataLoader(train_ds, batch_size=train_cfg.val_batch_size, shuffle=False, num_workers=train_cfg.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.val_batch_size, shuffle=False, num_workers=train_cfg.num_workers,
                             pin_memory=True)
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
    # load weights
    # model_weights_path = '../outputs/Audioset33--pretrain--V1.0/BZ-Audioset-pretrain33--Epoch08--BEST--model.pkl'
    model_weights_path = get_best_model_from_path(exp_dir)
    info("Loading model...")
    audio_model, loss = ModelManager.checkpoint_static_loader(model_weights_path, audio_model)
    info(f"Model loaded, loss = {loss} ")
    # print run config
    info(f"Using the following configs: BS = {train_cfg.val_batch_size}, N_workers{train_cfg.num_workers}")
    # run validations
    info("Checking train metrics:")
    train_metrics = validate_clustering_pretrained_model(audio_model, train_loader, device=train_cfg.device)
    info("Train metrics: ")
    print(train_metrics)
    info("Checking test metrics:")
    test_metrics = validate_clustering_pretrained_model(audio_model, test_loader, device=train_cfg.device)
    info("Test metrics: ")
    print(test_metrics)



