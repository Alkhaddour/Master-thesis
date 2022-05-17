"""
TODO: This file ha not been tested after refactoring
"""
import os
import sys


sys.path.append('../..')
from utils.display_utils import info
from utils.model_utils import get_best_model_from_path

from utils.basic_utils import dump_pickle, load_pickle, make_valid_path, dump_json
from torch import optim, nn
from configs.global_config import OUTPUT_PATH
from configs.data_configs import get_esc_config_v1 as esc_data_config
from configs.train_configs import get_config_v1 as train_config
from models.transformer import TransformerSED
from data_processing.datasets import ESC50Dataset
from utils.managers import ModelManager, TrainManager
from torch.utils.data import DataLoader
from dotted_dict import DottedDict

def load_weights(fold_exp_dir, audio_model):
    # load weights
    model_weights_path = os.path.join(fold_exp_dir, get_best_model_from_path(fold_exp_dir))
    info("Loading model...")
    audio_model, loss = ModelManager.checkpoint_static_loader(model_weights_path, audio_model)
    info(f"Model loaded, loss = {loss} ")
    return audio_model
    
# pretrained settings 
# -------------------
# load pretraining settings
pre_train_exp_dir = os.path.join(OUTPUT_PATH, 'exp_1/dim128_heads8_depth3_mlpdim1024_lr0.0001_(F--PCA_512_to_32_C--64_A--K-MEANS)')
pre_train_configs = load_pickle(os.path.join(pre_train_exp_dir, 'configs.pkl'))
# pre_tr_data_cfg = DottedDict(pre_train_configs['tr_data_cfg'])
# te_data_cfg = DottedDict(configs['te_data_cfg'])
pre_train_cfg = DottedDict(pre_train_configs['train_cfg'])
pre_model_cfg = DottedDict(pre_train_configs['model_cfg'])


# ESC settings
train_cfg = train_config({'n_epochs': 100, 
                            'patience': 30, 
                            'n_print_steps': 50,  
                            'batch_size': 128, 
                            'val_batch_size' :128, 
                            'device': 'cuda:1', 
                            'lr':1e-4})
esc_data_cfg = esc_data_config({'num_mel_bins': 128,
                               'target_length': 128,
                               'targets': 'ESC-10',
                               'n_class': 10})
exp_dir =  os.path.join(OUTPUT_PATH, f'exp_1/ESC{esc_data_cfg.n_class}--FineTune')
model_name = 'BZ-ESC{}'.format(esc_data_cfg.n_class)
make_valid_path(exp_dir, is_dir=True, exist_ok=True)

configs = { 'data_cfg': esc_data_cfg,
            'model_cfg': pre_model_cfg,
            'train_cfg': train_cfg}
json_out = os.path.join(exp_dir, 'configs.json')
pkl_out = os.path.join(exp_dir,  'configs.pkl')
info(f"Saving JSON configs to {json_out}")
dump_json(configs, outfile=json_out)
dump_pickle(configs, outfile=pkl_out)


for fold in range(1, 6):
    fold_exp_dir = os.path.join(exp_dir, str(fold))
    make_valid_path(fold_exp_dir, is_dir=True, exist_ok=True)

    train_ds = ESC50Dataset(root_dir=esc_data_cfg.root_dir,
                            metadata=esc_data_cfg.metadata,
                            audio_cfg=esc_data_cfg,
                            test_fold=fold,
                            mode='train',
                            esc10_only=True if esc_data_cfg.targets == 'ESC-10' else False,
                            virtual_size=esc_data_cfg.vsize)
    test_ds = ESC50Dataset(root_dir=esc_data_cfg.root_dir,
                           metadata=esc_data_cfg.metadata,
                           audio_cfg=esc_data_cfg,
                           test_fold=fold,
                           mode='test',
                           esc10_only=True if esc_data_cfg.targets == 'ESC-10' else False,
                           virtual_size=esc_data_cfg.vsize)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=train_cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers,
                             pin_memory=True)

    model_mgr = ModelManager(model_name, fold_exp_dir, progress_log_train=train_cfg.n_print_steps,
                             progress_log_val=train_cfg.n_print_steps, patience=train_cfg.patience)

    # laod weights
    audio_model = TransformerSED(n_mels=esc_data_cfg.num_mel_bins, 
                                    num_classes=pre_model_cfg.out, 
                                    dim=pre_model_cfg.dim, 
                                    depth=pre_model_cfg.depth, 
                                    heads=pre_model_cfg.heads,  
                                    mlp_dim=pre_model_cfg.mlp_dim, 
                                    dim_head=pre_model_cfg.dim_head,
                                    dropout = pre_model_cfg.dropout,
                                    emb_dropout = pre_model_cfg.emb_dropout
                                )

    audio_model = load_weights(pre_train_exp_dir, audio_model)
    audio_model = TransformerSED.re_init_last_layer(audio_model, out=esc_data_cfg.n_class)

    # change last layer to predictions

    optimizer = optim.Adam(audio_model.parameters(), lr=train_cfg.lr)
    loss_func = nn.CrossEntropyLoss()

    scheduler = None
    train_manager = TrainManager(n_epochs=train_cfg.n_epochs,
                                 loss_fn=loss_func,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 device=train_cfg.device,
                                 model_manager=model_mgr,
                                 verbose=False)

    train_manager.train_model(audio_model, train_loader=train_loader, val_loader=test_loader, train_fun=None)
