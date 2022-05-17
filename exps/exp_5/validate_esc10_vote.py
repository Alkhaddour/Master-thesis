"""
Validate best model we got in our experiment
"""
import os 
import sys

import pandas as pd

sys.path.append('../../')
from models.FeedForward import DenseN

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing.datasets import  ESC10_openL3_Dataset2, ESC5_openL3_Dataset2
from utils.managers import ModelManager
from utils.display_utils import info
from utils.model_utils import Metrics, get_best_model_from_path
from dotted_dict import DottedDict
from utils.basic_utils import load_pickle
from configs.global_config import OUTPUT_PATH, PROJECT_DIR

failed = []
def validate_vote(model, dataset, device, n_classes, return_acc=True, mode='train'):
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    for i in range(dataset.__len__()):
        emb, target, fname, event = dataset.__getitem__(i)
        # Find probs
        output = model(emb.to(device)) 
        # find argmax
        output = torch.argmax(output, dim=1)     
        # append to list
        y_true.append(target.item())
        y_pred.append(torch.mode(output)[0].item())
        if mode == 'test':
            if target.item() != torch.mode(output)[0].item():
                failed.append((fname, event, torch.mode(output)[0].item()))

    avg_metrics = Metrics.calculate_multi_class_metrics(y_true,y_pred, n_classes=n_classes,average='macro')   
    if return_acc:
        return avg_metrics['accuracy']                                                                             
    else:
        return avg_metrics

if __name__ == "__main__":
    # mdl_path = f'exp_5(esc)/in512--l2--dims[1024, 512]--drp0.0--do10--lr0.001'  # AVG Train Accuracy: 94.3 -- AVG Test Accuracy:  85.4 / Vote accuracy(98.4/91.7)
    # mdl_path =   f'exp_5(esc)_aug(2)/in512--l1--dims[512]--drp0.3--do10--lr0.001'
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------
    # mdl_path = f'exp_5(esc)/in512--l2--dims[512, 512]--drp[0.0, 0.1]--do10--lr0.001' # test acc: 92.50
    # mdl_path = f'exp_6(regular_esc10)/in512--l1--dims[512]--drp0.3--do10--lr0.001'
    # mdl_path = f'exp_6(inverse_esc10)/in512--l1--dims[512]--drp0.0--do10--lr0.001'

    # mdl_path = f'exp_5(esc10_H)/in512--l2--dims[512, 512]--drp[0.1, 0.0]--do5--lr0.001'
    # mdl_path = f'exp_5(esc10_P)/in512--l2--dims[1024, 512]--drp[0.1, 0.0]--do5--lr0.001'
    # mdl_path = f'exp_6(inverse_esc10_H)/in512--l1--dims[512]--drp0.1--do5--lr0.001'
    # mdl_path = f'exp_6(inverse_esc10_P)/in512--l1--dims[512]--drp0.1--do5--lr0.001'
    # mdl_path = f'exp_6(regular_esc10_H)/in512--l1--dims[1024]--drp0.1--do5--lr0.001'
    # mdl_path = f'exp_6(regular_esc10_P)/in512--l1--dims[1024]--drp0.1--do5--lr0.001'

    all_exps_path = 'exp_5(esc)'
    csv_id ='esc10'    # csv meta file mark: ('esc10', 'esc10_P', 'esc10_H')
    part = '' # pickle file:        ('_HP_regular', '_HP_inverse', '')
    n_classes= 10         # number of classes:  (5, 10)

    best_te_acc = -1
    best_tr_acc = -1
    file = None
    for mdl_path in os.listdir(os.path.join(OUTPUT_PATH, all_exps_path)):
        mdl_path= os.path.join(all_exps_path,mdl_path)
        if os.path.isdir(os.path.join(OUTPUT_PATH, mdl_path)) == False:
            continue
        
        #------------------------------------------------------------------------------------------------
        avg_tr_runs = 0.0
        avg_te_runs = 0.0
        runs_cnt = 1
        for i in range(runs_cnt):

            exp_dir = os.path.join(OUTPUT_PATH, mdl_path)
            avg_trn_acc = 0.0
            avg_tst_acc = 0.0
            for fold in range(1, 6):
            # Exp path to evaluate
                fold_exp_dir = os.path.join(exp_dir, str(fold))

                # load config from file
                load_config_file = True
                if load_config_file:
                    configs = load_pickle(os.path.join(exp_dir, 'configs.pkl'))
                    train_cfg = DottedDict(configs['train_cfg'])
                    model_cfg = DottedDict(configs['model_cfg'])
                else:
                    raise Exception("Unimplemented file")
                
                train_ds = ESC10_openL3_Dataset2(root_dir=f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512{part}',
                                                metadata=f'/home/alhasan/workdir/SED/data/ESC-50/{csv_id}.csv',
                                                test_fold=fold,
                                                mode='train',
                                                virtual_size='all')
                test_ds =  ESC10_openL3_Dataset2(root_dir=f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512{part}',
                                                metadata=f'/home/alhasan/workdir/SED/data/ESC-50/{csv_id}.csv',
                                                test_fold=fold,
                                                mode='test',
                                                virtual_size='all')
                # train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)
                # test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)

                # define model
                test_device = 'cuda:1' #train_cfg.device
                dense_model = DenseN(in_dim=model_cfg.in_dim, n_hidden_layers=model_cfg.n_hidden_layers, hidden_dims=model_cfg.hidden_dims, 
                                    out_dim=model_cfg.out_dim, dropout=model_cfg.dropout, return_logits=False).to(train_cfg.device)
                # load weights
                model_weights_path = get_best_model_from_path(fold_exp_dir,last=False) 
                info("Loading model...")
                audio_model, loss = ModelManager.checkpoint_static_loader(model_weights_path, dense_model)
                info(f"Model loaded, loss = {loss} ")
                # print run config
                # run validations
                train_acc = validate_vote(audio_model, train_ds, device=test_device,n_classes=n_classes, return_acc=True)
                avg_trn_acc += train_acc
                test_acc = validate_vote(audio_model, test_ds, device=test_device, n_classes=n_classes, return_acc=True, mode='test')
                avg_tst_acc += test_acc
            
            info(f"Train Accuracy: {avg_trn_acc/5*100:.1f}")
            info(f"Test Accuracy:  {avg_tst_acc/5*100:.1f}")

            avg_tr_runs += avg_trn_acc/5
            avg_te_runs += avg_tst_acc/5
        
        info(f"AVG Train Accuracy: {avg_tr_runs/runs_cnt*100:.1f}")
        info(f"AVG Test Accuracy:  {avg_te_runs/runs_cnt*100:.1f}")

        if avg_te_runs/runs_cnt > best_te_acc:
            best_te_acc = avg_te_runs/runs_cnt
            best_tr_acc = avg_tr_runs/runs_cnt
            file = mdl_path
    
    print("------------------------------------------------------------------------")
    print(f"Best model summary:")
    print(f"Train acc = {best_tr_acc*100:.2f}")
    print(f"Test acc  = {best_te_acc*100:.2f}")
    print(f"model:      {file}")
    print("------------------------------------------------------------------------")




# print("failed: ", failed)

# pd.DataFrame(failed, columns =['file', 'event', 'predicted']).to_csv('failed.csv')
