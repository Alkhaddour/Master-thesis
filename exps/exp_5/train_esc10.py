# Train ESC_scratch (55112)
import os
from re import I
import sys
from torch.utils.data import DataLoader
from torch import optim, nn



sys.path.append('../..')
from configs.global_config import OUTPUT_PATH
from models.FeedForward import DenseN
from data_processing.datasets import ESC10_openL3_Dataset, ESC5_openL3_Dataset
from utils.managers import ModelManager, TrainManager
from utils.basic_utils import dump_json, dump_pickle, make_valid_path
from utils.display_utils import info
from dotted_dict import DottedDict
from utils.managers import ModelManager


def train_esc(model_cfg, lr=1e-3):
    train_cfg= DottedDict({'batch_size': 128, 'num_workers': 16, 'device': device, 'lr': lr, 'n_print_steps': 500, 'n_epochs':100,'patience':50 })
    data_cfg = DottedDict({'targets': 'ESC-10', 'n_class': out_dim})

    name = f"in{model_cfg['in_dim']}--l{model_cfg['n_hidden_layers']}--dims{model_cfg['hidden_dims']}--drp{model_cfg['dropout']}--do{model_cfg['out_dim']}--lr{train_cfg.lr}"

    exp_dir =  os.path.join(OUTPUT_PATH, f'exp_6({part}_{meta_mark})/{name}')
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


    print(data_cfg, '\n', train_cfg, '\n', model_cfg)

    avg_loss = 0.0
    for fold in range(1, 6):
        fold_exp_dir = os.path.join(exp_dir, str(fold))
        make_valid_path(fold_exp_dir, is_dir=True, exist_ok=True)

        train_ds = ESC5_openL3_Dataset(root_dir= f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512_HP_{part}', # (1) 55112 samples in ESC-10 / (2) 89483
                                        metadata= f'/home/alhasan/workdir/SED/data/ESC-50/{meta_mark}.csv',
                                        test_fold=fold,
                                        mode='train',
                                        virtual_size=160*5)
        test_ds =  ESC5_openL3_Dataset(root_dir= f'/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512_HP_{part}',
                                        metadata= f'/home/alhasan/workdir/SED/data/ESC-50/{meta_mark}.csv',
                                        test_fold=fold,
                                        mode='test',
                                        virtual_size=40*5)

        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)
        test_loader =  DataLoader(test_ds,  batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)

        model_mgr = ModelManager(model_name, fold_exp_dir, progress_log_train=train_cfg.n_print_steps,
                                progress_log_val=train_cfg.n_print_steps, patience=train_cfg.patience)

        audio_model = DenseN(in_dim=model_cfg.in_dim, n_hidden_layers=model_cfg.n_hidden_layers, hidden_dims=model_cfg.hidden_dims, 
                            out_dim=model_cfg.out_dim, dropout=model_cfg.dropout, return_logits=True).to(train_cfg.device)
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
        avg_loss += train_manager.model_manager.best_val_loss
    return avg_loss/5

if __name__ == "__main__":
    meta_mark = 'esc10_H'
    part = 'inverse'
    out_dim = 5
    lr = 1e-3
    device='cuda:1'
    layers_cfg = [
                (1, [512]),
                (1, [1024]),
                (2, [1024, 512]),
                (2, [512, 512]),
                (3, [1024, 1024, 512]),
                (3, [1024, 512, 256]),
                # (4, [1024, 1024, 512, 512]),
                # (4, [1024, 512, 512, 256])
                ]

    do_cfgs = { # 2-layer dropout config
                1: [0.0, 0.1, 0.2, 0.3],
                2: [0.0, 0.1, 0.2, [0.0, 0.1], 
                                   [0.1, 0.0]],
                3: [0.0, 0.1, 0.2, [0.1, 0.1, 0.0], 
                                   [0.2, 0.1, 0.0], 
                                   [0.1, 0.3, 0.1]],
                4: [0.0, 0.1, 0.2, [0.3, 0.2, 0.1, 0.0], 
                                   [0.1, 0.2, 0.1, 0.0], 
                                   [0.0, 0.3, 0.1, 0.0]]
            }
    for hl in layers_cfg:
        for d in do_cfgs[hl[0]]:
            model_cfg = DottedDict({'in_dim': 512, 'n_hidden_layers': hl[0], 'hidden_dims':hl[1],'out_dim':out_dim, 'dropout': d})
            loss = train_esc(model_cfg, lr=lr)
            with open(os.path.join(OUTPUT_PATH, f'exp_6({part}_{meta_mark})/dense_report.txt'), 'a') as f:
                f.write(f"{loss} -- {str(model_cfg)} >> lr= {lr}\n")

