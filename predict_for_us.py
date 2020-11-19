import torch
from torch.utils.data import Dataset
import json
import os
import librosa
import numpy as np
from signal_process import *
from glob import glob
import pickle
from torch.utils.data import DataLoader
import json
from models import *


class DroneDataset(Dataset):
    def __init__(self, sr=48000, parameters=None):
        '''
        label_option : {'all', 'angle', 'class'}
        method : {'raw_audio', 'spectrogram', 'logspectrogram', 'melspectrogram', 'logmelspectrogram', 
        'mfcc', 'stft', 'librosa_cqt', 'librosa_chroma_cqt', 'librosa_chroma_cens', 'librosa_rms', 
        'rainbowgram'}
        '''
        
        
        super(DroneDataset, self).__init__()
        
        self.sr = sr
        
        self.parameters = parameters

        
    def __len__(self):
        return len(glob('../gjk_dataset/*.pkl'))

    def __getitem__(self, idx):

        with open('../gjk_dataset/'+f"{str(idx).zfill(5)}.pkl", 'rb') as f:
            audio, label_angle, label_class = pickle.load(f)
            
        label_angle = torch.Tensor(label_angle)
        label_class = torch.Tensor(label_class)
        
        return audio, label_angle, label_class, idx


hyper = EasyDict({'lr' : 0.0003,
                    'max_epochs' :50,
                    'step_size' : 10,
                    'gamma' : 0.9,
                    'batch_size' : 32,
                    'test_batch_size' : 32,
                    'gpus' : [0],
                    'num_workers' : 128,
                    'auto_lr_find' : False,
                    'save_top_k' : 3,
                    'folder' : 'best_model',
                    'early_stopping' : True,
                    'patience' : 5
                    })

folder = 'model_ckpt'
hyper.folder = folder

###### angle #####
ckpt = 'angle/epoch=299_val_loss=0.1106'
model_name = 'My_Angle_Resnet'
print('start angle')
angle_model = get_model(model_name, hyper)
print()

###################
ckpt = 'class/epoch=312_val_loss=0.8434'
model_name = 'My_Class_Resnet'
print('start class')
class_model = get_model(model_name, hyper)

angle_model = angle_model.cuda()
class_model = class_model.cuda()

dataset = DroneDataset(parameters=angle_model.hparams)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=16)

results = []
for audio, angle_batch, class_batch, idx in dataloader:
    
    audio = audio.cuda()
    
    with torch.no_grad():
        angle_model.eval()
        class_model.eval()
        angle_out = angle_model(audio).detach().cpu().numpy()
        class_out = class_model(audio).detach().cpu().numpy()
        print(angle_out)
        print(class_out)
    break
#         for num, (ang, cls_) in enumerate(zip(angle_out, class_out)):
#             id_dict = {}
#             id_dict['id'] = int(id_[num].item())
#             id_dict['angle'] = inference_rule(ang)
#             id_dict['class'] = inference_rule(cls_)
#             results.append(id_dict)
        
# final_output = {}
# final_output['track3_results'] = results
    
    
# with open('t3_.json', 'w') as f:
#     json.dump(final_output, f, cls=NpEncoder)

# print(final_output)
# print('finish')