# Train ESC_scratch
import os
from re import I
import sys
from torch.utils.data import DataLoader
from torch import optim, nn



sys.path.append('../..')
from utils.model_utils import get_last_model_from_path
from configs.global_config import OUTPUT_PATH
from models.transformer import TransformerClassifier
from data_processing.datasets import ESC50_openL3_Dataset
from utils.managers import ModelManager, TrainManager
from utils.basic_utils import dump_json, dump_pickle, make_valid_path
from utils.display_utils import info
from dotted_dict import DottedDict
from utils.managers import ModelManager
def train_last_only(model):
    model.performerSED.performer.eval()
    model.performerSED.linear1.train()
    return model

data_cfg = DottedDict({'targets': 'ESC-10', 'n_class': 10})
train_cfg= DottedDict({'batch_size': 128, 'num_workers': 16, 'device': 'cuda:1', 'lr': 1e-3, 'n_print_steps': 4000, 'n_epochs':100,'patience':50 })
model_cfg = DottedDict({'dim': 512, 'depth': 4, 'heads': 12,'dim_head':4, 'mlp_dim': 1024, 'dropout':0.5, 'out': data_cfg.n_class, 'pre_out':30 })

exp_dir =  os.path.join(OUTPUT_PATH, f'exp_2/ESC{data_cfg.n_class}--Scratch')
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

chkpt_path = get_last_model_from_path('/home/alhasan/workdir/SED/outputs/exp_2/__trash__/in512--d4--h12--dh4--dm1024--do30(2)')

print(data_cfg, '\n', train_cfg, '\n', model_cfg)
for fold in range(1, 6):
    fold_exp_dir = os.path.join(exp_dir, str(fold))
    make_valid_path(fold_exp_dir, is_dir=True, exist_ok=True)

    train_ds = ESC50_openL3_Dataset(root_dir='/home/alhasan/workdir/SED/data/ESC-50/esc(audio_16k)_openl3_embeddings_500_512',
                                    metadata='/home/alhasan/workdir/SED/data/ESC-50/esc50.csv',
                                    test_fold=fold,
                                    mode='train',
                                    esc10_only=True,
                                    virtual_size='all')
    test_ds =  ESC50_openL3_Dataset(root_dir='/home/alhasan/workdir/SED/data/ESC-50/esc(audio_16k)_openl3_embeddings_500_512',
                                    metadata='/home/alhasan/workdir/SED/data/ESC-50/esc50.csv',
                                    test_fold=fold,
                                    mode='test',
                                    esc10_only=True,
                                    virtual_size='all')

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=train_cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers,
                             pin_memory=True)

    model_mgr = ModelManager(model_name, fold_exp_dir, progress_log_train=train_cfg.n_print_steps,
                             progress_log_val=train_cfg.n_print_steps, patience=train_cfg.patience)

    audio_model = TransformerClassifier(dim=model_cfg.dim, depth=model_cfg.depth, heads=model_cfg.heads, dim_head=model_cfg.dim_head, 
                                        mlp_dim=model_cfg.mlp_dim, dropout=model_cfg.dropout, num_classes=model_cfg.pre_out)

    audio_model, loss = ModelManager.checkpoint_static_loader(chkpt_path, audio_model)
    info(f"Model loaded loss = {loss}")
    audio_model = TransformerClassifier.re_init_last_layer(audio_model, model_cfg.out).to(train_cfg.device)
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
    train_manager.train_model(audio_model, train_loader=train_loader, val_loader=test_loader, train_fun=None)#TransformerClassifier.train_last_only)
