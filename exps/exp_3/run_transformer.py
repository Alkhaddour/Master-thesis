import os, sys


sys.path.append('../../')
from torch import nn
from data_processing.datasets import PANNsOfflineDataset2
from torch.utils.data import DataLoader
from dotted_dict import DottedDict
from torch import optim
from models.transformer import TransformerClassifier
from configs.global_config import OUTPUT_PATH 
from utils.basic_utils import dump_json, dump_pickle, make_valid_path
from utils.display_utils import info
from utils.managers import ModelManager, TrainManager



# define config
train_cfg= DottedDict({'batch_size': 128, 'num_workers': 16, 'device': 'cuda:0', 'lr': 1e-5, 'n_print_steps': 4000, 'n_epochs':100,'patience':10 })
# load dataset
train_ds = PANNsOfflineDataset2(panns_dateset_pickle_file= '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_20_512_panns(classes_3_train).pkl', balanced_sampling=True)
test_ds = PANNsOfflineDataset2(panns_dateset_pickle_file= '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_20_512_panns(classes_3_test).pkl', balanced_sampling=True)

train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)

def run_transformer(model_cfg=None):
    # model_cfg = DottedDict({'dim': 512, 'depth': 2, 'heads': 4,'dim_head':4, 'mlp_dim': 1024, 'dropout':0.0, 'out': 30 })
    # define model
    transformer = TransformerClassifier(dim=model_cfg.dim, depth=model_cfg.depth, heads=model_cfg.heads, dim_head=model_cfg.dim_head, 
                                        mlp_dim=model_cfg.mlp_dim, dropout=model_cfg.dropout, num_classes=model_cfg.out)

    optimizer = optim.Adam(transformer.parameters(), lr=train_cfg.lr)
    loss_func = nn.CrossEntropyLoss()
    scheduler = None


    # define output directories
    name = f"in{model_cfg['dim']}--d{model_cfg['depth']}--h{model_cfg['heads']}--dh{model_cfg['dim_head']}--dm{model_cfg['mlp_dim']}--drp{model_cfg['dropout']}--do{model_cfg['out']}"
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
    train_mgr.train_model(transformer, train_loader=train_loader, val_loader=test_loader)
    info (f"Done! best loss: {model_mgr.best_val_loss}")
    return model_mgr.best_val_loss



if __name__ == "__main__":
    model_cfg = DottedDict({'dim': 512, 'depth': 3, 'heads': 8,'dim_head':64, 'mlp_dim': 512, 'dropout':0.1, 'out': 3})
    loss = run_transformer(model_cfg)
    # for depth in [3,2]:
    #     for heads in [8, 4]:
    #         for mlp_dim in [1024, 512]:
    #             for do in [0.0, 0.1, 0.4]:
    #                 model_cfg = DottedDict({'dim': 512, 'depth': depth, 'heads': heads,'dim_head':64, 'mlp_dim': mlp_dim, 'dropout':do, 'out': 30 })
    #                 loss = run_transformer(model_cfg)
    #                 with open(os.path.join(OUTPUT_PATH, 'exp_3/transformer_report.txt'), 'a') as f:
    #                     f.write(f"{str(model_cfg)} > {loss}\n")

