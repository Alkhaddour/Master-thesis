"""
Validate best model we got in our experiment
"""
import os 
import sys
from cv2 import threshold
import numpy as np

import pandas as pd
from prometheus_client import Metric

sys.path.append('../../')
from models.FeedForward import DenseN

import torch
import torch.nn.functional as F
from data_processing.datasets import  ESC10_openL3_Dataset2, ESC5_openL3_Dataset2
from utils.managers import ModelManager
from utils.display_utils import info
from utils.model_utils import Metrics, get_best_model_from_path
from configs.global_config import OUTPUT_PATH, PROJECT_DIR


#
# ------------ Validate
def validate_1(model_h, model_p, dataset_h, dataset_p):
    # Train accuracy: 80.31
    # Test accuracy:  70.75
    true_labels = []
    pred_labels = []

    for i in range(dataset_h.__len__()):
        emb, target, _, _ = dataset_h.__getitem__(i)
        # Find probs
        output_h = torch.sigmoid(model_h(emb.to(device))) # B x 5
        output_p = torch.sigmoid(model_p(emb.to(device))) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        preds = torch.argmax(probs, axis=1)
        pred_lbl = torch.mode(preds)[0].item()
        true_labels.append(target)
        pred_labels.append(pred_lbl)

    for i in range(dataset_p.__len__()):
        emb, target, _, _ = dataset_p.__getitem__(i)
        # Find probs
        output_h = torch.sigmoid(model_h(emb.to(device))) # B x 5
        output_p = torch.sigmoid(model_p(emb.to(device))) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        preds = torch.argmax(probs, axis=1)
        pred_lbl = torch.mode(preds)[0].item()
        true_labels.append(target+5)
        pred_labels.append(pred_lbl)

    avg_metrics = Metrics.calculate_multi_class_metrics(true_labels, pred_labels, n_classes=10,average='macro')   
    return avg_metrics['accuracy']    

def get_thresholds(model_h, model_p, dataset_h, dataset_p):
    true_labels = []
    pred_labels = []

    for i in range(dataset_h.__len__()):
        emb, target, _, _ = dataset_h.__getitem__(i)
        # Find probs
        output_h = torch.sigmoid(model_h(emb.to(device))) # B x 5
        output_p = torch.sigmoid(model_p(emb.to(device))) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        true_labels = true_labels + ([target] * probs.shape[0])
        pred_labels.append(probs)

    for i in range(dataset_p.__len__()):
        emb, target, _, _ = dataset_p.__getitem__(i)
        # Find probs
        output_h = torch.sigmoid(model_h(emb.to(device))) # B x 5
        output_p = torch.sigmoid(model_p(emb.to(device))) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        true_labels = true_labels + [target + 5] * probs.shape[0]
        pred_labels.append(probs)
    pred_labels = torch.cat(pred_labels).detach().cpu().numpy()
    thresholds = []
    for i in range(10):
        y_true = [int(x == i) for x in true_labels]
        thrs, _ = Metrics.get_best_prec_recall_threshold(y_true, pred_labels[:, i])
        thresholds.append(thrs)
    return thresholds 

def predict(model_h, model_p, dataset_h, dataset_p, thrs=None):
    true_labels = []
    pred_logits = []

    for i in range(dataset_h.__len__()):
        emb, target, _, _ = dataset_h.__getitem__(i)
        # Find probs
        output_h = model_h(emb.to(device)) # B x 5
        output_p = model_p(emb.to(device)) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        true_labels = true_labels + ([target] * probs.shape[0])
        pred_logits.append(probs)

    for i in range(dataset_p.__len__()):
        emb, target, _, _ = dataset_p.__getitem__(i)
        # Find probs
        output_h = model_h(emb.to(device)) # B x 5
        output_p = model_p(emb.to(device)) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        true_labels = true_labels + [target + 5] * probs.shape[0]
        pred_logits.append(probs)
    pred_logits = torch.cat(pred_logits).detach().cpu().numpy()
    true_labels = F.one_hot(torch.tensor(true_labels), num_classes=10).numpy()
    avg_metrics, _, thrs = Metrics.calculate_multi_label_metrics(torch.tensor(true_labels), torch.tensor(pred_logits), thrs=thrs)
    return avg_metrics['Accuracy'], thrs


def predict2(model_h, model_p, dataset_h, dataset_p, thresholds):
    true_labels = []
    pred_labels = []

    for i in range(dataset_h.__len__()):
        emb, target, _, _ = dataset_h.__getitem__(i)
        # Find probs
        output_h = torch.sigmoid(model_h(emb.to(device))) # B x 5
        output_p = torch.sigmoid(model_p(emb.to(device))) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        true_labels = true_labels + ([target] * probs.shape[0])
        pred_labels.append(probs)

    for i in range(dataset_p.__len__()):
        emb, target, _, _ = dataset_p.__getitem__(i)
        # Find probs
        output_h = torch.sigmoid(model_h(emb.to(device))) # B x 5
        output_p = torch.sigmoid(model_p(emb.to(device))) # B x 5
        probs = torch.cat([output_h, output_p], axis=1)
        true_labels = true_labels + [target + 5] * probs.shape[0]
        pred_labels.append(probs)
    pred_labels = torch.cat(pred_labels).detach().cpu().numpy()
    assert len(thresholds) == pred_labels.shape[1]
    pred_labels = np.array(pred_labels > thresholds, dtype=int)
    true_labels = F.one_hot(torch.tensor(true_labels), num_classes=10).numpy()
    assert pred_labels.shape == true_labels.shape, f"Expected preds and true to be of same size got {pred_labels.shape} and {true_labels.shape}"
    raise


    

# ----------- Params --------------
train_accs = 0.0
test_accs  = 0.0

for fold in range(1,6):
    print(f"processing fold {fold}/5]")
    ECS10_H_workdir = os.path.join(OUTPUT_PATH, 'exp_5(esc10_H)/in512--l2--dims[512, 512]--drp0.2--do5--lr0.001')               #2.1MB
    ECS10_P_workdir = os.path.join(OUTPUT_PATH, 'exp_5(esc10_P)/in512--l3--dims[1024, 512, 256]--drp0.0--do5--lr0.001')         #4.7MB


    device = 'cuda:1' #train_cfg.device
    h_model = DenseN(in_dim=512, n_hidden_layers=2, hidden_dims=[512, 512], out_dim=5, dropout=0.2, return_logits=True).to(device)
    p_model = DenseN(in_dim=512, n_hidden_layers=3, hidden_dims=[1024, 512, 256], out_dim=5, dropout=0.0, return_logits=True).to(device)

    # load weights
    h_model_path = get_best_model_from_path(os.path.join(ECS10_H_workdir, str(fold)),last=False) 
    p_model_path = get_best_model_from_path(os.path.join(ECS10_P_workdir, str(fold)),last=False) 

    h_model, h_loss = ModelManager.checkpoint_static_loader(h_model_path, h_model)
    p_model, h_loss = ModelManager.checkpoint_static_loader(p_model_path, p_model)

    preds_train = {'filename': {'true': None, 'HP':[]}}
    preds_test =  {'filename': {'true': None, 'HP':[]}}



    part = ''           # pickle file:        ('_HP_regular', '_HP_inverse', '')
    train_h = ESC5_openL3_Dataset2(root_dir=f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512',
                                    metadata=f'/home/alhasan/workdir/SED/data/ESC-50/esc10_H.csv',
                                    test_fold=fold,
                                    mode='train',
                                    virtual_size='all')
    train_p = ESC5_openL3_Dataset2(root_dir=f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512',
                                    metadata=f'/home/alhasan/workdir/SED/data/ESC-50/esc10_P.csv',
                                    test_fold=fold,
                                    mode='train',
                                    virtual_size='all')

    test_h =  ESC5_openL3_Dataset2(root_dir=f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512',
                                    metadata=f'/home/alhasan/workdir/SED/data/ESC-50/esc10_H.csv',
                                    test_fold=fold,
                                    mode='test',
                                    virtual_size='all')
    test_p =  ESC5_openL3_Dataset2(root_dir=f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512',
                                    metadata=f'/home/alhasan/workdir/SED/data/ESC-50/esc10_P.csv',
                                    test_fold=fold,
                                    mode='test',
                                    virtual_size='all')

    tr_accuracy, thrs = predict(h_model, p_model, train_h, train_p, thrs=None)        
    te_accuracy, _ = predict(h_model, p_model, test_h, test_p, thrs)            
    train_accs +=  tr_accuracy                            
    test_accs  +=  te_accuracy

print(f"Avg Train accuracy: {train_accs/5:.2f}")
print(f"Avg Test accuracy:  {test_accs/5:.2f}")

