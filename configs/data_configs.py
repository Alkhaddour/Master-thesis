from dotted_dict import DottedDict
import os 
from configs.global_config import PROJECT_DIR


def get_esc_config_v1(custom_cfg=None):
    cfg = DottedDict()
    cfg.freqm = 24  # maximum possible length of the mask.
    cfg.timem = 9*16000  # maximum possible length of the mask. (5000ms)
    cfg.vsize = 'all'
    cfg.num_mel_bins = 128
    cfg.target_length = 512
    cfg.mode = 'train'
    cfg.norms = [-6.6268077, 5.358466]
    cfg.skip_norm = False
    cfg.root_dir = os.path.join(PROJECT_DIR, 'data/ESC-50/raw/audio_16k')
    cfg.metadata =  os.path.join(PROJECT_DIR, 'data/ESC-50/esc50.csv')
    cfg.targets = 'ESC-10'
    if cfg.targets == 'ESC-10':
        cfg.n_class = 10
    elif cfg.targs == 'ESC-50':
        cfg.n_class = 10
    else:
        raise Exception("Unknown target set")

    # finally, load custom configs
    if custom_cfg is not None:
        for k, v in custom_cfg.items():
            cfg[k] = v
    return cfg


def get_audioset_config_v1(custom_cfg=None):
    cfg = DottedDict()
    cfg.freqm = 24  # maximum possible length of the mask.
    cfg.timem = 5*16000  # maximum possible length of the mask. (5000ms)
    cfg.vsize = 'all'
    cfg.num_mel_bins = 128
    cfg.target_length = 512
    cfg.norms = [-4.2677393, 4.5689974]
    cfg.skip_norm = False
    cfg.root_dir = '../data/AudioSet/raw'
    cfg.meta_path = None  # '../data/AudioSet/audioset_meta_train.csv'
    cfg.mid_name_map_file = '../data/AudioSet/class_labels_indices.csv'
    cfg.targets = ['Baby cry, infant cry', 'Chainsaw', 'Chicken, rooster', 'Clock', 'Dog', 'Firecracker', 'Helicopter',
                   'Ocean', 'Rain', 'Rain on surface', 'Raindrop', 'Sneeze', 'Waves, surf', 'Cat', 'Cheering',
                   'Clapping', 'Cough', 'Cowbell', 'Engine', 'Frog', 'Glass', 'Keys jangling', 'Meow', 'Music', 'Pig',
                   'Scratch', 'Squeal', 'Tap', 'Vehicle', 'Walk, footsteps', 'Water tap, faucet', 'Wild animals',
                   'Wind']
    if cfg.targets == 'all':
        cfg.n_class = 527
    else:
        cfg.n_class = len(cfg.targets)

    # finally, load custom configs
    if custom_cfg is not None:
        for k, v in custom_cfg.items():
            cfg[k] = v
    return cfg

def get_clustering_config_v1(custom_cfg=None):
    cfg = DottedDict()
    cfg.freqm = 24  # maximum possible length of the mask.
    cfg.timem = 3200  # maximum possible length of the mask. (200ms)
    cfg.vsize = 'all'
    cfg.num_mel_bins = 128
    cfg.target_length = 128
    cfg.norms = [-4.2677393, 4.5689974]
    cfg.skip_norm = False
    cfg.root_dir = os.path.join(PROJECT_DIR, '/data/AudioSet/raw')
    cfg.meta_path = None
    cfg.n_class = 128

    # finally, load custom configs
    if custom_cfg is not None:
        for k, v in custom_cfg.items():
            cfg[k] = v
    return cfg