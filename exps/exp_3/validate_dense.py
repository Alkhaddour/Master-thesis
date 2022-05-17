"""
Validate best model we got in our experiment
"""
import os 
import sys

sys.path.append('../../')
from models.FeedForward import DenseN

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing.datasets import  PANNsOfflineDataset2
from utils.managers import ModelManager
from utils.display_utils import info
from utils.model_utils import Metrics, get_best_model_from_path
from dotted_dict import DottedDict
from utils.basic_utils import load_pickle
from configs.global_config import OUTPUT_PATH, PROJECT_DIR


def validate_clustering_pretrained_model(model, data_loader, device):
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

    avg_metrics = Metrics.calculate_multi_class_metrics(torch.hstack(y_true),torch.hstack(y_pred), n_classes=3,average='macro')                                                                                
    return avg_metrics

if __name__ == "__main__":
    # Exp path to evaluate
    exp_dir = os.path.join(OUTPUT_PATH, 'exp_4/in512--l2--dims[1024, 512]--drp0.3--do3--lr0.0001')
    # load config from file
    load_config_file = True
    if load_config_file:
        configs = load_pickle(os.path.join(exp_dir, 'configs.pkl'))
        train_cfg = DottedDict(configs['train_cfg'])
        model_cfg = DottedDict(configs['model_cfg'])
    else:
        raise Exception("Unimplemented file")
    
    root_dir = os.path.join(PROJECT_DIR, 'data/AudioSet/raw')
    train_ds = PANNsOfflineDataset2(panns_dateset_pickle_file= '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_20_512_panns(classes_3r_train).pkl', balanced_sampling=False)
    test_ds = PANNsOfflineDataset2(panns_dateset_pickle_file= '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_20_512_panns(classes_3r_test).pkl', balanced_sampling=False)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)

    # define model
    test_device = 'cuda:1' #train_cfg.device
    dense_model = DenseN(in_dim=model_cfg.in_dim, n_hidden_layers=model_cfg.n_hidden_layers, hidden_dims=model_cfg.hidden_dims, 
                         out_dim=model_cfg.out_dim, dropout=model_cfg.dropout, return_logits=True).to(test_device)
    # load weights
    # model_weights_path = '../outputs/Audioset33--pretrain--V1.0/BZ-Audioset-pretrain33--Epoch08--BEST--model.pkl'
    model_weights_path = get_best_model_from_path(exp_dir) # '/home/alhasan/workdir/SED/outputs/exp_4/in512--l2--dims[1024, 512]--drp0.3--do3--lr0.0001/panns2.3--Epoch12--TMP--model.pkl' 
    info("Loading model...")
    audio_model, loss = ModelManager.checkpoint_static_loader(model_weights_path, dense_model)
    info(f"Model loaded, loss = {loss} ")
    # print run config
    # run validations
    info("Checking train metrics:")
    train_metrics = validate_clustering_pretrained_model(audio_model, train_loader, device=test_device)
    info("Train metrics: ")
    print(train_metrics)
    info("Checking test metrics:")
    test_metrics = validate_clustering_pretrained_model(audio_model, test_loader, device=test_device)
    info("Test metrics: ")
    print(test_metrics)




