from dotted_dict import DottedDict


def get_config_v1(custom_cfg=None):
    cfg = DottedDict()
    cfg.batch_size = 32
    cfg.val_batch_size = 14
    cfg.lr = 5e-5
    cfg.n_epochs = 30#10
    cfg.patience = 5
    # system params
    cfg.num_workers = 16
    cfg.device = 'cuda'
    # display params
    cfg.n_print_steps = 80
    # finally, load custom configs
    if custom_cfg is not None:
        for k, v in custom_cfg.items():
            cfg[k] = v
    return cfg