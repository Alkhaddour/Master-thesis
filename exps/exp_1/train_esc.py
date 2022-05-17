#Train ESC_scratch

import os
import sys
from torch.utils.data import DataLoader
from torch import optim, nn

sys.path.append('../..')

from configs.global_config import OUTPUT_PATH
from configs.data_configs import get_esc_config_v1 as esc_data_config
from configs.train_configs import get_config_v1 as train_config
from configs.model_configs import get_performer_config_v1 as model_config
from models.transformer import TransformerSED
from data_processing.datasets import ESC50Dataset
from utils.managers import ModelManager, TrainManager
from utils.basic_utils import dump_json, dump_pickle, make_valid_path
from utils.display_utils import info


data_cfg = esc_data_config({'num_mel_bins': 128,
                               'target_length': 128,
                               'targets': 'ESC-10',
                               'n_class': 10})
train_cfg = train_config({'n_epochs': 100, 
                            'patience': 30, 
                            'n_print_steps': 50,  
                            'batch_size': 128, 
                            'val_batch_size' :128, 
                            'device': 'cuda:1', 
                            'lr':1e-3})
model_cfg = model_config({'dim':128, 
                            'depth':3, 
                            'heads':8, 
                            'mlp_dim': 1024,
                            'dim_head': 64,
                            'out':10,
                            'dropout': 0.1,
                            'emb_dropout': 0.01})

exp_dir =  os.path.join(OUTPUT_PATH, f'exp_1/ESC{data_cfg.n_class}--Scratch')
make_valid_path(exp_dir, is_dir=True, exist_ok=True)

model_name = 'BZ-ESC{}'.format(data_cfg.n_class)

configs = { 'data_cfg': data_cfg,
            'model_cfg': model_cfg,
            'train_cfg': train_cfg}
json_out = os.path.join(exp_dir, 'configs.json')
pkl_out = os.path.join(exp_dir,  'configs.pkl')
info(f"Saving JSON configs to {json_out}")
dump_json(configs, outfile=json_out)
dump_pickle(configs, outfile=pkl_out)
for fold in range(1, 6):
    fold_exp_dir = os.path.join(exp_dir, str(fold))
    make_valid_path(fold_exp_dir, is_dir=True, exist_ok=True)

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
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers,
                             pin_memory=True)

    model_mgr = ModelManager(model_name, fold_exp_dir, progress_log_train=train_cfg.n_print_steps,
                             progress_log_val=train_cfg.n_print_steps, patience=train_cfg.patience)

    audio_model = TransformerSED(n_mels=data_cfg.num_mel_bins, 
                                    num_classes=model_cfg.out, 
                                    dim=model_cfg.dim, 
                                    depth=model_cfg.depth, 
                                    heads=model_cfg.heads,  
                                    mlp_dim=model_cfg.mlp_dim, 
                                    dim_head=model_cfg.dim_head,
                                    dropout = model_cfg.dropout,
                                    emb_dropout = model_cfg.emb_dropout
                                )
                               
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
    train_manager.train_model(audio_model, train_loader=train_loader, val_loader=test_loader)
