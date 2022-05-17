import sys
sys.path.append('../')


import random
import torchaudio
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional
from torch.utils.data import Dataset
import random
import pandas as pd
import os
from utils.basic_utils import csv_to_dict, load_pickle
from utils.display_utils import info
import torchaudio.transforms as T
from tqdm import tqdm
from utils.exceptions import TooShortAudio
import gc

def load_features(splits_root_dir, features_meta_path):
    files =[]
    chunk_idx =[]
    features = []

    dataset = FeaturesDastset(splits_root_dir, features_meta_path)
    loader = DataLoader(dataset, batch_size=16, num_workers=12)
    for x,y,z in loader:
        files=files + list(x)
        chunk_idx.append(y)
        features.append(torch.reshape(z, (-1, 512)))
    features = torch.vstack(features)
    features = features[features.sum(dim=1) != 1024]
    chunk_idx = torch.hstack(chunk_idx)
    chunk_idx = [x.item() for x in chunk_idx] # convert tensor to elements

    return files, chunk_idx, features

class AudiosetDataset(Dataset):
    def __init__(self, data_cfg, mode):
        """
        init Audioset dataset
        :param root_dir: root dir to audio data which is the parent dir to paths in metafile
        :param meta_path: path to csv file with metadata (path to audio, classes within the audio)
        :param mid_name_map_file: path to csv with map between class readable name and its code
        :param data_cfg: dict of all configs for features extraction
        :param targets: list of classes (human-readable) used in classification or 'all' to use all classes.
        """
        self.root_dir = data_cfg.root_dir
        self.meta = pd.read_csv(data_cfg.meta_path)
        self.class_id_map = pd.read_csv(data_cfg.mid_name_map_file)
        self.data_cfg = data_cfg
        self.target_classes = data_cfg.targets
        self.virtual_size = data_cfg.vsize
        self.mode = mode
        # Create mapping between classes and their indices
        self.mid_name_map, self.name_mid_map = csv_to_dict(df=self.class_id_map, key_col='mid',
                                                           val_col='display_name', two_ways=True)

        if isinstance(self.target_classes, list):
            # use classes in list and remove extra labels indicated in meta['classes']
            self.meta = self.meta.loc[self.meta['labels'].isin(self.target_classes)]
            self.target_mids = [self.name_mid_map[name] for name in self.target_classes]
        else:
            assert self.target_classes == 'all', f"targets must be list of classes or str 'all"
            self.target_mids = [self.name_mid_map[name] for name in list(self.meta['classes'])]

        self.samples_path = list(self.meta['audio_path'])
        self.samples_class = list(self.meta['classes'])

        # features extraction
        self.melbins = self.data_cfg.num_mel_bins
        self.norm_mean = self.data_cfg.norms[0]
        self.norm_std = self.data_cfg.norms[1]
        self.skip_norm = self.data_cfg.skip_norm

    def _encode_classes(self, sample_classes):
        code = np.zeros(len(self.target_classes))
        for s_class in sample_classes.split(","):
            if s_class in self.target_mids:
                idx = self.target_mids.index(s_class)
                code[idx] = 1.0
        return code

    def _wav2fmelspec(self, filename):
        # load file
        waveform, sr = torchaudio.load(filename, normalize=True)
        try:
            waveform = waveform[0, :]  # take single channel
        except Exception as e:
            info(f"Invalid file: {filename} -- raised: {str(e)}")
        assert sr == 16000, f"Expected sr = 16Khz got {sr}"
        # extract filter banks
        transform = T.MelSpectrogram(sample_rate=sr,
                                     n_fft=1024,  # Num of points used to find FFT
                                     win_length=400,  # 25ms when sr 16kHz
                                     hop_length=160,  # 10ms when sr 16kHz
                                     pad=0,  # default value
                                     n_mels=self.melbins,
                                     power=2.0,
                                     normalized=False,
                                     center=True,
                                     pad_mode='reflect',
                                     onesided=True,
                                     norm=None,
                                     mel_scale='htk')
        mel_specgram = transform(waveform)  # INPUT: (signal) OUTPUT: (Bins, Time)
        # fit to input size
        target_length = self.data_cfg.get('target_length')
        n_frames = mel_specgram.shape[1]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, p, 0, 0))
            mel_specgram = m(mel_specgram)
        elif p < 0:
            mel_specgram = mel_specgram[:, 0:target_length]
        return mel_specgram

    def _extract_features(self, filename) -> torch.Tensor:
        # Extract fbank
        mel_specgram = self._wav2fmelspec(filename)
        # Normalize the input for both training and test
        if not self.skip_norm:
            mel_specgram = (mel_specgram - self.norm_mean) / (self.norm_std * 2)

        return mel_specgram

    def __len__(self):
        if self.virtual_size == 'all':
            return len(self.meta)
        else:
            return self.virtual_size

    def __getitem__(self, index):
        # load sample from self.samples_path[index]
        # index = random.randint(0, self.__len__() - 1)
        sample_path = os.path.join(self.root_dir, self.samples_path[index])
        # process audio

        try:
            features = self._extract_features(sample_path)
        except Exception as e:
            print(f"Failed to use file: {sample_path}, because of the following exception: \n{e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # augment train data
        if self.mode == 'train':
            p = random.uniform(0, 1)
            if p < 0.33:
                features = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.data_cfg.freqm)(features)
            elif p < 0.66:
                features = torchaudio.transforms.TimeMasking(time_mask_param=self.data_cfg.timem)(features)
            # else keep the original signal

        # create one-hot encoding for classes
        label = self._encode_classes(self.samples_class[index])
        # return sample and encoding as tensor
        return features, torch.tensor(label)

class ESC50Dataset(Dataset):
    def __init__(self, root_dir, metadata, audio_cfg, mode, test_fold, esc10_only=True, virtual_size=500):
        """
        init ESC-50 dataset
        :param root_dir: root dir to audio data which is the parent dir to paths in metafile
        :param metadata: path to csv with map between file, class readable name and its fold
        :param audio_cfg: dict of all configs for features extraction
        :param targets: list of classes (human-readable) used in classification or 'all' to use all classes.
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(metadata)
        self.mode = mode
        if esc10_only == True:
            self.meta = self.meta.loc[self.meta['esc10'] == True]
        # take certain fold
        if self.mode == 'train':
            self.meta = self.meta.loc[self.meta['fold'] != test_fold]
        elif self.mode == 'test':
            self.meta = self.meta.loc[self.meta['fold'] == test_fold]
        # extract filename and class
        self.samples_path = list(self.meta['filename'])
        self.samples_class = list(self.meta['category'])
        self.data_cfg = audio_cfg
        self.virtual_size = virtual_size

        self.target_classes = list(self.meta['category'].unique())

    def _encode_classes(self, sample_class):
        # code = np.zeros(len(self.target_classes))
        idx = self.target_classes.index(sample_class)
        # code[idx] = 1
        # return code
        return idx

    def _wav2fmelspec(self, filename):
        # load file
        waveform, sr = torchaudio.load(filename, normalize=True)
        try:
            waveform = waveform[0, :]  # take single channel
        except Exception as e:
            info(f"Invalid file: {filename} -- raised: {str(e)}")
        assert sr == 16000, f"Expected sr = 16Khz got {sr}"
        # extract filter banks
        transform = T.MelSpectrogram(sample_rate=sr,
                                     n_fft=1024,  # Num of points used to find FFT
                                     win_length=400,  # 25ms when sr 16kHz
                                     hop_length=160,  # 10ms when sr 16kHz
                                     pad=0,  # default value
                                     n_mels=self.data_cfg.num_mel_bins,
                                     power=2.0,
                                     normalized=False,
                                     center=True,
                                     pad_mode='reflect',
                                     onesided=True,
                                     norm=None,
                                     mel_scale='htk')
        mel_specgram = transform(waveform)  # INPUT: (signal) OUTPUT: (Bins, Time)
        # fit to input size
        target_length = self.data_cfg.get('target_length')
        n_frames = mel_specgram.shape[1]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, p, 0, 0))
            mel_specgram = m(mel_specgram)
        elif p < 0:
            mel_specgram = mel_specgram[:, 0:target_length]
        return mel_specgram

    def _extract_features(self, filename) -> torch.Tensor:
        # Extract fbank
        mel_specgram = self._wav2fmelspec(filename)
        # Normalize the input for both training and test
        if not self.data_cfg.skip_norm:
            mel_specgram = (mel_specgram - self.data_cfg.norms[0]) / (self.data_cfg.norms[1] * 2)

        return mel_specgram

    def __len__(self):
        if self.virtual_size == 'all':
            return len(self.meta)
        else:
            return self.virtual_size

    def __getitem__(self, index):
        # load sample from self.samples_path[index]
        # index = random.randint(0, self.__len__() - 1)
        sample_path = os.path.join(self.root_dir, self.samples_path[index])
        # process audio
        try:
            features = self._extract_features(sample_path)
        except Exception as e:
            print(f"Failed to use file: {sample_path}, because of the following exception: \n{e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # augment train data
        if self.mode == 'train':
            p = random.uniform(0, 1)
            if p < 0.33:
                features = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.data_cfg.freqm)(features)
            elif p < 0.66:
                features = torchaudio.transforms.TimeMasking(time_mask_param=self.data_cfg.timem)(features)
            # else keep the original signal

        # create one-hot encoding for classes
        label = self._encode_classes(self.samples_class[index])
        # return sample and encoding as tensor
        return features, torch.tensor(label)

class FeaturesDastset(Dataset):
    def __init__(self, splits_root_dir, features_meta_path, verbose=True):
        self.splits_root_dir = splits_root_dir
        self.features_meta_path = features_meta_path
        self.features_meta = pd.read_csv(self.features_meta_path)
        self.verbose = verbose
        info(f"Dataset initialized, # files = {self.__len__()}")

    def __len__(self):
        return len(self.features_meta)

    def _pad(self, embs):
        emb_shape = embs.shape[1]
        batch_size = embs.shape[0]
        if batch_size < 11:
            padding =  np.ones((11-batch_size,emb_shape)) * 2
            return np.append(embs,padding, axis=0)
        else:
            return embs
        

    def __getitem__(self, idx):
        audio_name = self.features_meta.iloc[idx]['audio_name'][:-4]
        audio_name += '.wav'

        filename = os.path.join(self.splits_root_dir, self.features_meta.iloc[idx]['audio_name'])
        n_splits= self.features_meta.iloc[idx]['n_splits']
        embs = np.load(filename)

        return audio_name, n_splits, self._pad(embs)

class ClusteringDataset(Dataset):
    def __init__(self,root_dir, meta_file, data_cfg, mode=None):
        # set attributes
        self.root_dir = root_dir
        self.meta_data = pd.read_csv(meta_file)
        # from config
        self.data_cfg = data_cfg
        self.melbins = self.data_cfg.num_mel_bins
        self.norm_mean = self.data_cfg.norms[0]
        self.norm_std = self.data_cfg.norms[1]
        self.skip_norm = self.data_cfg.skip_norm
        # preprocess meta
        assert self.data_cfg.n_class == self.get_lables_count(), f"Expected n_class = {self.data_cfg.n_class} got {self.get_lables_count()} in {os.path.basename(meta_file)}"
        self.mode = mode
        info(f"Dataset initialized... len dataset = {self.__len__()}")
    
    def get_lables_count(self):
        return len(list(self.meta_data['clustering_labels'].unique()))

    def __len__(self):
        return len(self.meta_data)

    def _wav2fmelspec(self, filename, sample_ts):
        # load file
        waveform, sr = torchaudio.load(filename, normalize=True)
        # check sample rate
        assert sr == 16000, f"Expected sr = 16Khz got {sr}"
        # isolate chunk
        fst = sample_ts * sr
        lst = (sample_ts+1)* sr


        waveform = waveform[0, fst:lst]  # take single channel
        if waveform.size()[0] < sr//8:
            raise TooShortAudio(f"Very short segment of size {waveform.size()[0]}")

        # extract filter banks
        transform = T.MelSpectrogram(sample_rate=sr,
                                     n_fft=1024,  # Num of points used to find FFT
                                     win_length=400,  # 25ms when sr 16kHz
                                     hop_length=160,  # 10ms when sr 16kHz
                                     pad=0,  # default value
                                     n_mels=self.melbins,
                                     power=2.0,
                                     normalized=False,
                                     center=True,
                                     pad_mode='reflect',
                                     onesided=True,
                                     norm=None,
                                     mel_scale='htk')
        mel_specgram = transform(waveform)  # INPUT: (signal) OUTPUT: (Bins, Time)
        return mel_specgram

    def _crop_pad(self, mel_specgram, target_length):
        # fit to input size
        n_frames = mel_specgram.shape[1]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, p, 0, 0))
            mel_specgram = m(mel_specgram)
        elif p < 0:
            max_start = mel_specgram.shape[1] - target_length
            start = random.randint(0, max_start)
            mel_specgram = mel_specgram[:, start:start+target_length]
        return mel_specgram

    def _extract_features(self, filename, sample_ts) -> torch.Tensor:
        # Extract fbank
        mel_specgram = self._wav2fmelspec(filename, sample_ts)
        # Cropping and Padding
        mel_specgram = self._crop_pad(mel_specgram, target_length = self.data_cfg.get('target_length'))
        # Normalize the input for both training and test
        if not self.skip_norm:
            mel_specgram = (mel_specgram - self.norm_mean) / (self.norm_std * 2)

        return mel_specgram
    
    def _encode_classes(self, sample_class):
        """
        we don't need it here as we use CrossEntropyLoss which require index of class instead fo one hot encoding
        """
        code = np.zeros(self.get_lables_count())
        code[sample_class] = 1
        return code

    def __getitem__(self, index):
        sample_path = os.path.join(self.root_dir, self.meta_data.iloc[index]['files'])
        sample_ts =  self.meta_data.iloc[index]['chucks']
        
        # process audio
        try:
            features = self._extract_features(sample_path, sample_ts)
        except Exception as e:
            if isinstance(e, TooShortAudio):
                pass
            else:
                info(f"Failed to use file: {sample_path} @ sample {sample_ts}, because of the following exception: \n{e}")            
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # augment train data
        if self.mode == 'train':
            p = random.uniform(0, 1)
            if p < 0.33:
                features = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.data_cfg.freqm)(features)
            elif p < 0.66:
                features = torchaudio.transforms.TimeMasking(time_mask_param=self.data_cfg.timem)(features)
            # else keep the original signal

        label = self.meta_data.iloc[index]['clustering_labels']
        # return sample and encoding as tensor
        return features, torch.tensor(label)

class PANNsOfflineDataset(Dataset):
    def __init__(self,embeddings_file, labels_file, balanced_sampling=True, files = None):
        # set attributes
        self.embeddings_file = embeddings_file
        self.labels_file = labels_file
        self.balanced_sampling=balanced_sampling

        # load data
        info("Loading files...")
        self.embeddings_dict = load_pickle(self.embeddings_file)['embeddings']
        self.labels_dict = load_pickle(self.labels_file)['labels']

        # load dataset data
        self.embeddings = []
        self.as_labels = []

        # Files
        if files is None:
            self.files = [f for f in self.embeddings_dict.keys() if f in self.labels_dict.keys()]
        else:
            self.files = files

        info("Collect data")
        for f in tqdm(self.files):
            n_segments = min(len(self.embeddings_dict[f]), len(self.labels_dict[f]))
            self.embeddings.append(self.embeddings_dict[f][:n_segments])
            self.as_labels.append(self.labels_dict[f][:n_segments])
        
        # concatenate
        info("Concatenate labels")
        self.as_labels = np.concatenate(self.as_labels, axis=0)
        info("Concatenate embeddings") # ~4 min
        self.embeddings = np.concatenate(self.embeddings, axis=0)

        # map labels
        self.unique_labels = list(set(self.as_labels))
        self.labels_map = {self.unique_labels[i]: i for i in range(len(self.unique_labels))}

        # class samples
        self.class_samples = {}
        for lbl in self.unique_labels:
            lbl_samples = [idx for idx in range(len(self.as_labels)) if self.as_labels[idx] == lbl]
            self.class_samples[self.labels_map[lbl]] = lbl_samples

        # print summary
        info("PANNs dataset ready...")
        info(f"Number of samples: {len(self.embeddings)}")
        info(f"Number of classes: {len(self.unique_labels)}")

        # free memory
        del self.embeddings_dict
        del self.labels_dict
        gc.collect()
        info("dataset initialized")

    def __len__(self):
        return len(self.as_labels)

    def sample_item_bal(self, idx):
        class_ = idx % len(self.unique_labels)
        sample_idx = random.choice(self.class_samples[class_])
        assert(class_ == self.labels_map[self.as_labels[sample_idx]])
        return  self.embeddings[sample_idx], class_

 
    def __getitem__(self, idx):
        if self.balanced_sampling:
            return self.sample_item_bal(idx)            
        else:
            return self.embeddings[idx], self.labels_map[self.as_labels[idx]]

class PANNsOfflineDataset2(Dataset):
    def __init__(self,panns_dateset_pickle_file, balanced_sampling=True):
        # set attributes
        self.panns_dateset_pickle_file = panns_dateset_pickle_file
        self.balanced_sampling=balanced_sampling

        # load data
        info("Loading files...")
        self.panns_dateset = load_pickle(self.panns_dateset_pickle_file)
        self.labels = self.panns_dateset['labels']
        self.embeddings = self.panns_dateset['embeddings']
         
        # map labels
        self.unique_labels = list(set(self.labels))
        self.labels_map = {self.unique_labels[i]: i for i in range(len(self.unique_labels))}
        print("labels map: ", self.labels_map)

        # class samples
        if balanced_sampling == True:
            info("Preparing for balanced sampling...")
            self.class_samples = {}
            for lbl in tqdm(self.unique_labels):
                lbl_samples = [idx for idx in range(len(self.labels)) if self.labels[idx] == lbl]
                self.class_samples[self.labels_map[lbl]] = lbl_samples
        else:
            info("Preparing for random sampling...")
            c = list(zip(self.labels, self.embeddings))
            random.shuffle(c)
            self.labels, self.embeddings = zip(*c)



        # print summary
        info("PANNs dataset ready...")
        info(f"Number of samples: {len(self.embeddings)}")
        info(f"Number of classes: {len(self.unique_labels)}")

        # free memory
        del self.panns_dateset
        gc.collect()
        info("dataset initialized")

    def __len__(self):
        return len(self.labels)

    def sample_item_bal(self, idx):
        class_ = idx % len(self.unique_labels)
        sample_idx = random.choice(self.class_samples[class_])
        assert(class_ == self.labels_map[self.labels[sample_idx]])
        return  self.embeddings[sample_idx], class_

 
    def __getitem__(self, idx):
        if self.balanced_sampling:
            return self.sample_item_bal(idx)            
        else:
            return self.embeddings[idx], self.labels_map[self.labels[idx]]

class ESC50_openL3_Dataset(Dataset):
    def __init__(self, root_dir, metadata, mode, test_fold, esc10_only=True, virtual_size='all'):
        """
        init ESC-50 dataset
        :param root_dir: root dir to audio data which is the parent dir to paths in metafile
        :param metadata: path to csv with map between file, class readable name and its fold
        :param audio_cfg: dict of all configs for features extraction
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(metadata)
        self.mode = mode
        if esc10_only == True:
            self.meta = self.meta.loc[self.meta['esc10'] == True]
        # take certain fold
        if self.mode == 'train':
            self.meta = self.meta.loc[self.meta['fold'] != test_fold]
        elif self.mode == 'test':
            self.meta = self.meta.loc[self.meta['fold'] == test_fold]
        # extract filename and class
        self.samples_path = list(self.meta['filename'])
        self.samples_class = list(self.meta['category'])
        self.virtual_size = virtual_size
        self.target_classes = list(self.meta['category'].unique())

    def _encode_classes(self, sample_class):
        # code = np.zeros(len(self.target_classes))
        idx = self.target_classes.index(sample_class)
        # code[idx] = 1
        # return code
        return idx

    def _extract_features(self, filename) -> torch.Tensor:
        # Extract fbank
        embs = np.load(filename)   
        idx_to_return=np.random.randint(0, len(embs))
        # Normalize the input for both training and test
        return embs[idx_to_return]

    def __len__(self):
        if self.virtual_size == 'all':
            return len(self.meta)
        else:
            return self.virtual_size

    def __getitem__(self, index):
        # load sample from self.samples_path[index]
        if index >= len(self.meta):
            index = index % len(self.meta)
        sample_path = os.path.join(self.root_dir, self.samples_path[index]+'.npy')
        # TODO: change file type to pickle

        # process audio
        try:
            features = self._extract_features(sample_path)
        except Exception as e:
            print(f"Failed to use file: {sample_path}, because of the following exception: \n{e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # create one-hot encoding for classes
        label = self._encode_classes(self.samples_class[index])
        # return sample and encoding as tensor
        return features, torch.tensor(label)

class ESC10_openL3_Dataset(Dataset):
    def __init__(self, root_dir, metadata, mode, test_fold, virtual_size='all'):
        """
        init ESC-50 dataset
        :param root_dir: root dir to audio data which is the parent dir to paths in metafile
        :param metadata: path to csv with map between file, class readable name and its fold
        :param audio_cfg: dict of all configs for features extraction
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(metadata)
        self.mode = mode
        # take certain fold
        if self.mode == 'train':
            self.meta = self.meta.loc[self.meta['fold'] != test_fold]
        elif self.mode == 'test':
            self.meta = self.meta.loc[self.meta['fold'] == test_fold]
        # extract filename and class
        self.samples_path = list(self.meta['filename'])
        self.events = list(self.meta['event'])
        self.virtual_size = virtual_size
        self.event_id = list(self.meta['event_id'])

    def _encode_classes(self, idx):
        return self.event_id[idx]

    def _extract_features(self, filename) -> torch.Tensor:
        # Extract fbank
        embs = np.load(filename)   
        idx_to_return=np.random.randint(0, len(embs))
        # Normalize the input for both training and test
        return embs[idx_to_return]

    def __len__(self):
        if self.virtual_size == 'all':
            return len(self.meta)
        else:
            return self.virtual_size

    def __getitem__(self, index):
        # load sample from self.samples_path[index]
        if index >= len(self.meta):
            index = index % len(self.meta)
        sample_path = os.path.join(self.root_dir, self.samples_path[index]+'.npy')
        # TODO: change file type to pickle

        # process audio
        try:
            features = self._extract_features(sample_path)
        except Exception as e:
            print(f"Failed to use file: {sample_path}, because of the following exception: \n{e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # create one-hot encoding for classes
        label = self._encode_classes(index)
        # return sample and encoding as tensor
        return features, torch.tensor(label)

class ESC5_openL3_Dataset(Dataset):
    def __init__(self, root_dir, metadata, mode, test_fold, virtual_size='all'):
        """
        init ESC-50 dataset
        :param root_dir: root dir to audio data which is the parent dir to paths in metafile
        :param metadata: path to csv with map between file, class readable name and its fold
        :param audio_cfg: dict of all configs for features extraction
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(metadata)
        self.mode = mode
        # take certain fold
        if self.mode == 'train':
            self.meta = self.meta.loc[self.meta['fold'] != test_fold]
        elif self.mode == 'test':
            self.meta = self.meta.loc[self.meta['fold'] == test_fold]
        # extract filename and class
        self.samples_path = list(self.meta['filename'])
        self.events = list(self.meta['event'])
        self.virtual_size = virtual_size
        self.event_id = list(self.meta['event_id_5'])

    def _encode_classes(self, idx):
        return self.event_id[idx]

    def _extract_features(self, filename) -> torch.Tensor:
        # Extract fbank
        embs = np.load(filename)   
        idx_to_return=np.random.randint(0, len(embs))
        # Normalize the input for both training and test
        return embs[idx_to_return]

    def __len__(self):
        if self.virtual_size == 'all':
            return len(self.meta)
        else:
            return self.virtual_size

    def __getitem__(self, index):
        # load sample from self.samples_path[index]
        if index >= len(self.meta):
            index = index % len(self.meta)
        sample_path = os.path.join(self.root_dir, self.samples_path[index]+'.npy')
        # TODO: change file type to pickle

        # process audio
        try:
            features = self._extract_features(sample_path)
        except Exception as e:
            print(f"Failed to use file: {sample_path}, because of the following exception: \n{e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # create one-hot encoding for classes
        label = self._encode_classes(index)
        # return sample and encoding as tensor
        return features, torch.tensor(label)

class ESC10_openL3_Dataset2(Dataset):
    def __init__(self, root_dir, metadata, mode, test_fold, virtual_size='all'):
        """
        init ESC-50 dataset
        :param root_dir: root dir to audio data which is the parent dir to paths in metafile
        :param metadata: path to csv with map between file, class readable name and its fold
        :param audio_cfg: dict of all configs for features extraction
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(metadata)
        self.mode = mode
        # take certain fold
        if self.mode == 'train':
            self.meta = self.meta.loc[self.meta['fold'] != test_fold]
        elif self.mode == 'test':
            self.meta = self.meta.loc[self.meta['fold'] == test_fold]
        # extract filename and class
        self.samples_path = list(self.meta['filename'])
        self.events = list(self.meta['event'])
        self.virtual_size = virtual_size
        self.event_id = list(self.meta['event_id'])

    def _encode_classes(self, idx):
        return self.event_id[idx]

    def _extract_features(self, filename) -> torch.Tensor:
        # Extract fbank
        embs = np.load(filename)   
        max_get = min(len(embs), 1000)
        # Normalize the input for both training and test
        return np.array([ embs[i] for i in random.sample(range(len(embs)), max_get) ])
    def __len__(self):
        if self.virtual_size == 'all':
            return len(self.meta)
        else:
            return self.virtual_size

    def __getitem__(self, index):
        # load sample from self.samples_path[index]
        if index >= len(self.meta):
            index = index % len(self.meta)
        sample_path = os.path.join(self.root_dir, self.samples_path[index]+'.npy')
        # TODO: change file type to pickle

        # process audio
        try:
            features = self._extract_features(sample_path)
        except Exception as e:
            print(f"Failed to use file: {sample_path}, because of the following exception: \n{e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # create one-hot encoding for classes
        label = self._encode_classes(index)
        # return sample and encoding as tensor
        return torch.tensor(features), torch.tensor(label), self.samples_path[index], self.events[index]


class ESC5_openL3_Dataset2(Dataset):
    def __init__(self, root_dir, metadata, mode, test_fold, virtual_size='all'):
        """
        init ESC-50 dataset
        :param root_dir: root dir to audio data which is the parent dir to paths in metafile
        :param metadata: path to csv with map between file, class readable name and its fold
        :param audio_cfg: dict of all configs for features extraction
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(metadata)
        self.mode = mode
        # take certain fold
        if self.mode == 'train':
            self.meta = self.meta.loc[self.meta['fold'] != test_fold]
        elif self.mode == 'test':
            self.meta = self.meta.loc[self.meta['fold'] == test_fold]
        # extract filename and class
        self.samples_path = list(self.meta['filename'])
        self.events = list(self.meta['event'])
        self.virtual_size = virtual_size
        self.event_id = list(self.meta['event_id_5'])

    def _encode_classes(self, idx):
        return self.event_id[idx]

    def _extract_features(self, filename) -> torch.Tensor:
        # Extract fbank
        embs = np.load(filename)   
        max_get = min(len(embs), 1000)
        # Normalize the input for both training and test
        return np.array([ embs[i] for i in random.sample(range(len(embs)), max_get) ])
    def __len__(self):
        if self.virtual_size == 'all':
            return len(self.meta)
        else:
            return self.virtual_size

    def __getitem__(self, index):
        # load sample from self.samples_path[index]
        if index >= len(self.meta):
            index = index % len(self.meta)
        sample_path = os.path.join(self.root_dir, self.samples_path[index]+'.npy')
        # TODO: change file type to pickle

        # process audio
        try:
            features = self._extract_features(sample_path)
        except Exception as e:
            print(f"Failed to use file: {sample_path}, because of the following exception: \n{e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # create one-hot encoding for classes
        label = self._encode_classes(index)
        # return sample and encoding as tensor
        return torch.tensor(features), torch.tensor(label), self.samples_path[index], self.events[index]

if __name__ == "__main__":
    # ds = PANNsOfflineDataset2(panns_dateset_pickle_file= '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_500_512_panns(classes_35).pkl', balanced_sampling=False)
    ds = ESC5_openL3_Dataset2(root_dir='/home/alhasan/workdir/SED/data/ESC-50/esc10(nosilence)_openl3_200_512',
                                metadata='/home/alhasan/workdir/SED/data/ESC-50/esc10_H.csv',
                                test_fold=1,
                                mode='test',
                                virtual_size='all')
    for i in range(ds.__len__()):
        emb, lbl, _, _= ds.__getitem__(i)
        print(f"Element ({i}): emb_shape = {emb.shape}, label = {lbl}")
