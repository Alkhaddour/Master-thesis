from dotted_dict import DottedDict


def get_performer_config_v1(custom_cfg=None):
    cfg = DottedDict()
    cfg.dim=512
    cfg.depth=4
    cfg.heads=4
    cfg.out=10

    # finally, load custom configs
    if custom_cfg is not None:
        for k, v in custom_cfg.items():
            cfg[k] = v
    return cfg

def get_transformer_config_v1(custom_cfg=None):
    cfg = DottedDict()
    cfg.dim=512
    cfg.depth=4
    cfg.heads=4
    cfg.mlp_dim = 512
    cfg.out=10

    # finally, load custom configs
    if custom_cfg is not None:
        for k, v in custom_cfg.items():
            cfg[k] = v
    return cfg