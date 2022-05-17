
import sys
from tabnanny import verbose

sys.path.append('../')
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import numpy as np
import pandas as pd
from configs.global_config import PANNS_PATH
from utils.display_utils import info
from statistics import mode

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

class Cnn14_DecisionLevelMax(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14_DecisionLevelMax, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32     # Downsampled ratio

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}

        return output_dict



class PANNs_labeler:
    def __init__(self, sr, ws,hs, mb, fmn, fmx, model_chkpt, device,labeling_accuracy=1.0, labeling_hop=0.5, min_coverage=0.7, labels=None, verbose=False):
        # init params
        self.sample_rate=sr
        self.window_size = ws
        self.hop_size = hs
        self.mel_bins = mb
        self.fmin =  fmn
        self.fmax = fmx
        self.model_type = 'Cnn14_DecisionLevelMax'
        self.classes_num = 527 # from audioset
        self.checkpoint_path = model_chkpt
        self.device = torch.device(device)
        self.labels = labels
        self.labeling_accuracy=labeling_accuracy  # averaging over 1 second audio
        self.labeling_hop=labeling_hop            # with hope of 0.5 hop
        self.min_coverage=min_coverage            # coverage within 1 second should be 70%
        self.verbose=verbose

        # init model
        Model = eval(self.model_type)
        self.model = Model(sample_rate=self.sample_rate, window_size=self.window_size, hop_size=self.hop_size, mel_bins=self.mel_bins, 
                      fmin=self.fmin, fmax=self.fmax, classes_num=self.classes_num)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])

        # Parallel
        info('GPU number: {}'.format(torch.cuda.device_count()), verbose=self.verbose)
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)
    
    def move_data_to_device(self, x, device):
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)
        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)
        else:
            return x

        return x.to(device)


    def extract_label(self, audio_path, ):
        """
        INPUT: path to audio file
        OUTPUT: top AudioSet class per segment
        """     
        # Load audio
        (waveform, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
        waveform = waveform[None, :]    # (1, audio_length)
        # info(f"wavelength: {(waveform.shape)}")
        waveform = self.move_data_to_device(waveform, self.device)

        # Forward
        with torch.no_grad():
            self.model.eval()
            batch_output_dict = self.model(waveform, None)

        framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
        """(time_steps, classes_num)"""

        info('Sound event detection result (time_steps x classes_num): {}'.format(framewise_output.shape), verbose=self.verbose)
        hard_output = np.argmax(framewise_output, axis=1)[:-1] # exclude last one as it is usually resulting from padding
        info('hard ouput result (time_steps x 1): {}'.format(hard_output.shape), verbose=self.verbose)

        return self.get_global_labels(hard_output, sig_len=waveform.shape[1])
        

    def extract_glabel_from_take(self, samples_in_take, min_coverage=0.9):
        mde = mode(samples_in_take)
        cnt_mde = len([x for x in samples_in_take if x == mde])
        coverage = cnt_mde / len(samples_in_take)
        if coverage >= min_coverage:
            return mde, coverage
        else:
            return None, coverage # No class in 
        

    def get_global_labels(self, flabels, sig_len):
        sample_time = 0.020 # (20ms)
        samples_per_take = int(self.labeling_accuracy / sample_time)
        samples_per_hop =  int(self.labeling_hop / sample_time)
        n_takes = int(int(sig_len / self.sample_rate) / self.labeling_hop) - 1
        # info(f"samples_per_take = {samples_per_take}")
        # info(f"samples_per_hop = {samples_per_hop}")
        # info(f"n_takes = {n_takes}")
        
        visited = 0
        g_labels =[]
        for i in range(0, len(flabels) - samples_per_take + 1, samples_per_hop):
            fst = i 
            lst = i + samples_per_take
            samples_in_take = flabels[fst:lst]
            assert len(samples_in_take) == samples_per_take, f"error at batch {i} expected {samples_per_take} samples in segment got {len(samples_in_take)}"
            g_label,pct = self.extract_glabel_from_take(samples_in_take, min_coverage=self.min_coverage)
            g_labels.append(g_label)
            # print(f">>>>> from {fst:5d}   to  {lst:5d}: label = {g_label} -- pct {pct}")

            visited += 1
        # info(f"Visited {visited} sampels")
        if self.labels != None:
            h_labels = [self.labels[int(x)] if x is not None else None for x in g_labels] # return human readable labels
        else:
            h_labels = None
        return g_labels, h_labels


if __name__ =='__main__':
    labels = pd.read_csv('../data/AudioSet/class_labels_indices.csv')['display_name'].to_list()
    labeler = PANNs_labeler(sr=16000, ws=1024,hs=320, mb=64, fmn=50, fmx=14000, model_chkpt=PANNS_PATH, device='cuda:0', labels=labels, verbose=True)
    g_labels, h_labels = labeler.extract_label('./6XOsBs2rZRg_30.wav')
    # print(frames_labels.shape)
    # print(h_labels.shape)
    print(g_labels)
    print(h_labels)

