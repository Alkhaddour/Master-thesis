import os
import sys
sys.path.append('../..')
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.basic_utils import load_pickle


from configs.global_config import OUTPUT_PATH
from models.transformer import TransformerSED
from data_processing.datasets import ESC50Dataset

from utils.managers import ModelManager
from utils.display_utils import info
from utils.model_utils import Metrics, get_best_model_from_path


def validate_pretrained_model(model, data_loader, device):
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    for step, (data, target) in tqdm(enumerate(data_loader)):
        # Find logits
        output = model(data.to(device))
        # append to list
        y_true.append(target.detach().cpu())
        y_pred.append(torch.argmax(output, dim=1).detach().cpu())

    avg_metrics = Metrics.calculate_multi_class_metrics(torch.hstack(y_true),
                                                        torch.hstack(y_pred),
                                                        n_classes=10,
                                                        average='macro')
    return avg_metrics


def load_weights(fold_exp_dir, audio_model):
    # load weights
    model_weights_path = os.path.join(fold_exp_dir, get_best_model_from_path(fold_exp_dir))
    info("Loading model...")
    audio_model, loss = ModelManager.checkpoint_static_loader(model_weights_path, audio_model)
    info(f"Model loaded, loss = {loss} ")
    return audio_model

if __name__ == "__main__":
    # define data loader
    validate_pretrained = True # set to true if you want to validate pretrained model
    exp_dir = os.path.join(OUTPUT_PATH,f'exp_1/ESC10--FineTune') # ESC10--Scratch
    # Load saved configs
    configs = load_pickle(os.path.join(exp_dir, 'configs.pkl'))
    data_cfg = configs['data_cfg']
    model_cfg = configs['model_cfg']
    train_cfg = configs['train_cfg']

    model_name = 'BZ-ESC{}'.format(data_cfg.n_class)
    total_train={}
    total_test={}
    for fold in range(1, 6):
        fold_exp_dir = os.path.join(exp_dir, str(fold))
        res_file = open(os.path.join(fold_exp_dir, 'results.txt'), 'w')
        train_ds = ESC50Dataset(root_dir=data_cfg.root_dir,
                                metadata=data_cfg.metadata,
                                audio_cfg=data_cfg,
                                test_fold=fold,
                                mode='train',
                                esc10_only=True if data_cfg.targets == 'ESC-10' else False,
                                virtual_size=data_cfg.vsize)
        test_ds = ESC50Dataset(root_dir=data_cfg.root_dir,
                               metadata=data_cfg.metadata,
                               audio_cfg=data_cfg,
                               test_fold=fold,
                               mode='test',
                               esc10_only=True if data_cfg.targets == 'ESC-10' else False,
                               virtual_size=data_cfg.vsize)

        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                                  num_workers=train_cfg.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True,
                                 num_workers=train_cfg.num_workers,
                                 pin_memory=True)
        
        audio_model = TransformerSED(n_mels=data_cfg.num_mel_bins, 
                                    num_classes=data_cfg.n_class, 
                                    dim=model_cfg.dim, 
                                    depth=model_cfg.depth, 
                                    heads=model_cfg.heads,  
                                    mlp_dim=model_cfg.mlp_dim, 
                                    dim_head=model_cfg.dim_head,
                                    dropout = model_cfg.dropout,
                                    emb_dropout = model_cfg.emb_dropout)

        audio_model = load_weights(fold_exp_dir, audio_model)
        # run validations
        info("Checking train metrics:")
        train_metrics = validate_pretrained_model(audio_model, train_loader, device=train_cfg.device)
        info("Train metrics: ")
        print(train_metrics)

        res_file.write("Train metrics:\n")
        for k, v in train_metrics.items():
            res_file.write(f"{k} = {v}\n")
            if k in total_train:
                total_train[k].append(v)
            else:
                total_train[k] = [v]

        info("Checking test metrics:")
        test_metrics = validate_pretrained_model(audio_model, test_loader, device=train_cfg.device)
        info("Test metrics: ")
        print(test_metrics)

        res_file.write("Test metrics:\n")
        for k, v in test_metrics.items():
            res_file.write(f"{k} = {v}\n")
            if k in total_test:
                total_test[k].append(v)
            else:
                total_test[k] = [v]
        res_file.close()

    # summarize results
    res_file = open(os.path.join(exp_dir, 'total_results.txt'), 'w')
    res_file.write(f"Train metrics\n")
    for k, v in total_train.items():
        res_file.write(f"avg {k} = {np.mean(v)}\n")
    res_file.write(f"\n")
    res_file.write(f"Test metrics\n")
    for k, v in total_test.items():
        res_file.write(f"avg {k} = {np.mean(v)}\n")
    res_file.close()



