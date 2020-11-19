import torch
from torch.utils.data import Dataset
import json
import os
import librosa
import numpy as np
from signal_process import *
from glob import glob
import pickle

        
        
class DroneDataset(Dataset):
    def __init__(self,audio_dir='sample_dataset', label_option='all', sr=48000, method='raw_audio', test=False, 
                 parameters=None, denoise=False, validation=False, datetype='logmel'):
        '''
        label_option : {'all', 'angle', 'class'}
        method : {'raw_audio', 'spectrogram', 'logspectrogram', 'melspectrogram', 'logmelspectrogram', 
        'mfcc', 'stft', 'librosa_cqt', 'librosa_chroma_cqt', 'librosa_chroma_cens', 'librosa_rms', 
        'rainbowgram'}
        '''
        
        assert label_option in ['all', 'angle', 'class']
        
        super(DroneDataset, self).__init__()
        self.label_option = label_option
        self.audio_dir = audio_dir
        self.sr = sr
        self.maxlen = sr * 30
        self.method = method
        self.test = test
        self.parameters = parameters
        self.denoise = denoise
        self.validation = validation
        self.type_ = datetype
    def __len__(self):
        if self.test:
            if self.audio_dir[-1] == '/':
                return len(glob(self.audio_dir+'*.wav'))
            else:
                return len(glob(self.audio_dir+'/'+'*.wav'))
        else:
            if self.validation:
                return len(glob('../yeop/dataset/validation_%s/*.pkl'%self.type_))
            else:
                return len(glob('../yeop/dataset/train_%s/*.pkl'%self.type_))

    def __getitem__(self, idx):
        if not self.test:
            if self.validation:
                with open('../yeop/dataset/validation_%s/%s.pkl'%(self.type_, idx), 'rb') as f:
                    audio, label_angle, label_class = pickle.load(f)
            else:
                with open('../yeop/dataset/train_%s/%s.pkl'%(self.type_, idx), 'rb') as f:
                    audio, label_angle, label_class = pickle.load(f)
        else:
            audio, sr = librosa.load(os.path.join(self.audio_dir, f"{str(idx).zfill(5)}.wav"), 
                                     sr=self.sr, mono=False)
        
            audio = np.pad(audio, ((0,0), (0,self.maxlen-len(audio[0]))), 'constant')
            self.parameters['sr'] = self.sr
            audio = signal_process(torch.tensor(audio), method=self.method, parameters=self.parameters,
                                denoise=self.denoise)
            
            return audio, idx
            

        label_angle = torch.Tensor(label_angle)
        label_class = torch.Tensor(label_class)
        
        if self.label_option == 'all':
            return audio, (label_angle, label_class)
        
        elif self.label_option == 'angle':
            return audio, label_angle
        else:
            return audio, label_class