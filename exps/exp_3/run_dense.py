import os, sys

sys.path.append('../../')
from torch import nn
from data_processing.datasets import PANNsOfflineDataset2
from torch.utils.data import DataLoader
from dotted_dict import DottedDict
from torch import optim
from models.FeedForward import DenseN
from configs.global_config import OUTPUT_PATH 
from utils.basic_utils import dump_json, dump_pickle, make_valid_path
from utils.display_utils import info
from utils.managers import ModelManager, TrainManager



# define config
train_cfg= DottedDict({'batch_size': 128, 'num_workers': 16, 'device': 'cuda:0', 'lr': 1e-4, 'n_print_steps': 500, 'n_epochs':100,'patience':10 })
# load dataset
train_ds = PANNsOfflineDataset2(panns_dateset_pickle_file= '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_20_512_panns(classes_3r_train).pkl', balanced_sampling=False)
test_ds = PANNsOfflineDataset2(panns_dateset_pickle_file= '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_20_512_panns(classes_3r_test).pkl', balanced_sampling=False)

train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)

def run_dense(model_cfg):
    # model_cfg = DottedDict({'in_dim': 512, 'n_hidden_layers': 4, 'hidden_dims':[1024,1024,512, 512],'out_dim':30, 'dropout': 0.1 })
    # define model
    dense = DenseN(in_dim=model_cfg.in_dim, n_hidden_layers=model_cfg.n_hidden_layers, hidden_dims=model_cfg.hidden_dims, 
                out_dim=model_cfg.out_dim, dropout=model_cfg.dropout, return_logits=True).to(train_cfg.device)


    optimizer = optim.Adam(dense.parameters(), lr=train_cfg.lr)
    loss_func = nn.CrossEntropyLoss()
    scheduler = None


    # define output directories
    name = f"in{model_cfg['in_dim']}--l{model_cfg['n_hidden_layers']}--dims{model_cfg['hidden_dims']}--drp{model_cfg['dropout']}--do{model_cfg['out_dim']}--lr{train_cfg.lr}"
    exp_prefix = f'exp_4/{name}'
    exp_dir = os.path.join(OUTPUT_PATH, exp_prefix)
    model_name = 'panns2.3'
    make_valid_path(exp_dir, is_dir=True, exist_ok=True)

    # save configs
    configs = { 'train_cfg': train_cfg,
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
    train_mgr.train_model(dense, train_loader=train_loader, val_loader=test_loader)
    info (f"Done! best loss: {model_mgr.best_val_loss}")
    return model_mgr.best_val_loss



if __name__ == "__main__":
    layers_cfg = [
                # (1, [512]),
                (2, [1024, 512]),
                # (2, [512, 512]),
                #   (3, [1024, 1024, 512]),
                #   (3, [1024, 512, 256]),
                #   (4, [1024, 1024, 512, 512]),
                #   (4, [1024, 512, 512, 256])
                ]

    do_cfgs = { # 2-layer dropout config
                1: [0.5],
                2: [0.3, 0.1, [0.0, 0.1], 
                              [0.1, 0.0]],
                3: [0.0, 0.1, [0.1, 0.1, 0.0], 
                              [0.2, 0.1, 0.0], 
                              [0.1, 0.3, 0.1]],
                # 4: [0.0, 0.1, [0.3, 0.2, 0.1, 0.0], 
                #               [0.1, 0.2, 0.1, 0.0], 
                #               [0.0, 0.3, 0.1, 0.0]]
            }
    for hl in layers_cfg:
        for d in do_cfgs[hl[0]]:
            model_cfg = DottedDict({'in_dim': 512, 'n_hidden_layers': hl[0], 'hidden_dims':hl[1],'out_dim':3, 'dropout': d})
            loss = run_dense(model_cfg)
            with open(os.path.join(OUTPUT_PATH, 'exp_4/dense_report.txt'), 'a') as f:
                f.write(f"{str(model_cfg)} > {loss}")


            