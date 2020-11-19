import torchaudio
import torch
import librosa 
import numpy as np
import noisereduce as nr
#from rainbowgram.wave_rain import wave2rain
'''
설명 = 오디오 데이터 preprocessing method
Args:
    audio_samples  : tensor  = 변환하려는 데이터
    method         : string  = method   ex: spectrogram
    parameters     : dict    = 변환하려는 method에 필요한 parameter들
Returns
    Tensor
'''
def signal_process(audio_samples, method='spectrogram', parameters=None, denoise=True):
    if denoise:
        left_channel, right_channel = noise_reduce(audio_samples, parameters)
        left_channel = torch.tensor(left_channel.astype(np.float32))
        right_channel = torch.tensor(right_channel.astype(np.float32))
    else:
        left_channel = audio_samples[0]
        right_channel = audio_samples[1]
    
    if method == 'raw_audio':
        output = audio_samples
    elif method == 'spectrogram': 
        spec_layer = torchaudio.transforms.Spectrogram(n_fft=parameters['n_fft'], win_length=parameters['win_length'], hop_length=parameters['hop_length'], power=parameters['power'], normalized=parameters['normalized'])
        spec = torch.stack([spec_layer(left_channel), spec_layer(right_channel)], dim=0)
        output = spec
    elif method == 'logspectrogram':
        spec_layer = torchaudio.transforms.Spectrogram(n_fft=parameters['n_fft'], win_length=parameters['win_length'], hop_length=parameters['hop_length'], power=parameters['power'], normalized=parameters['normalized'])
        spec = torch.stack([spec_layer(left_channel), spec_layer(right_channel)], dim=0)
        log_spec= torch.log(torch.clamp(spec, min=1e-6))
        output = log_spec
    elif method == 'melspectrogram':
        mel_layer = torchaudio.transforms.MelSpectrogram(n_fft=parameters['n_fft'], win_length=parameters['win_length'], hop_length=parameters['hop_length'], sample_rate=parameters['sample_rate'], f_min=parameters['f_min'], f_max=parameters['f_max'], n_mels=parameters['n_mels'])
        mel_spec = torch.stack([mel_layer(left_channel), mel_layer(right_channel)], dim=0)
        output = mel_spec
    elif method == 'logmelspectrogram':
        mel_layer = torchaudio.transforms.MelSpectrogram(n_fft=parameters['n_fft'], win_length=parameters['win_length'], hop_length=parameters['hop_length'], sample_rate=parameters['sample_rate'], f_min=parameters['f_min'], f_max=parameters['f_max'], n_mels=parameters['n_mels'])
        mel_spec = torch.stack([mel_layer(left_channel), mel_layer(right_channel)], dim=0)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))
        output = log_mel
    elif method == 'mfcc':
        melkwargs = {'hop_length': 1024, 'n_mels': 128}
        mfcc_layer = torchaudio.transforms.MFCC(sample_rate=parameters['sr'], n_mfcc=40, melkwargs=melkwargs)
        output = mfcc_layer(audio_samples)
    
    else:
        raise NotImplementedError
    
    return normalize(output)

# 설명  = Instantaneous frequency를 표현한것
#     Args:
#         wave (numpy array [T,]): time-domain waveform
#         sr:       Sampling Rate.
#         n_fft:    FFT length.
#         stride:   Pop length (default 3/4 overlap).
#         power:    Strength coefficient (1:magnitude, 2:power).
#         clip:     Magnitude clipping boarder (<=clip become clip).
#         log_mag:  Flag whether use log magnitude or not.
#         range:    
#         mel_freq: Flag whether use mel-frequency for not.
#     Returns:
#         numpy.ndarray n_fft/2+1 x frame: rainbowgram
    # elif method == 'rainbowgram':
    #     f = list()
    #     for i in range(len(audio_samples)):
    #         f.append(wave2rain(audio_samples[i], sr=parameters['sr'], n_fft=parameters['n_fft'], stride=parameters['hop_length'], power=parameters['power'], clip=parameters['clip'], log_mag=parameters['log_mag'], mel_freq=parameters['mel_freq']))
    #     return torch.Tensor(f)
    
def noise_reduce(audio_samples, parameters=None):
    c1, c2 = stereo_to_mono(audio_samples, trim=False, numpy=True)
    reduced_noise_c1 = nr.reduce_noise(audio_clip=c1, noise_clip=c2, verbose=False, n_grad_freq=1, n_fft=parameters['n_fft'], win_length=parameters['win_length'], hop_length=parameters['hop_length'], n_std_thresh=parameters['n_std_thresh'])
    reduced_noise_c2 = nr.reduce_noise(audio_clip=c2, noise_clip=c1, verbose=False, n_grad_freq=1, n_fft=parameters['n_fft'], win_length=parameters['win_length'], hop_length=parameters['hop_length'], n_std_thresh=parameters['n_std_thresh'])
    return reduced_noise_c1, reduced_noise_c2
'''
설명 = stereo 데이터를 2개 채널로 각각 return
audio  : tensor   = tensor 형태의 원 데이터
trim   : bool     = 30초로 늘어난 데이터 다시 원 데이터 길이로 줄이기 
numpy  : bool     = return type Numpy or Tensor
return : tuple    = (tensor, tensor) / (np, np)
'''
def stereo_to_mono(audio, trim=False, numpy=False):
    c1 = audio[0].numpy()
    c2 = audio[1].numpy()
    if trim:
        c1 = np.trim_zeros(c1)
        c2 = np.trim_zeros(c2)
        if c1.shape[0] >= c2.shape[0]:
            c1 = c1[:c2.shape[0]]
        else:
            c2 = c2[:c1.shape[0]]
    if not numpy:
        return torch.Tensor(c1), torch.Tensor(c2)
    return c1, c2
'''
설명 = stereo 데이터를 2개의 채널로 분리하여 signal processing 해주는 method
audio      : tensor   = tensor 형태의 원 데이터
method     : string   = processing 전처리 method. ex: logmelspectrogram
param_dict : dict     = 변환할때 바꿔야되는 parameter. ex: n_fft, n_mels, win_size, hop_length
trim       : bool     = 30초로 늘어난 데이터 다시 원 데이터 길이로 줄이기 
numpy      : bool     = return type Numpy or Tensor
return     : tuple    = (tensor, tensor) / (np, np)
'''
def stereo_to_mono_processing(audio, method, parameters, trim=False, numpy=False):
    c1, c2 = stereo_to_mono(audio, trim=trim, numpy=numpy)
    left_spec = signal_process(c1, method=method, parameters=parameters)
    right_spec = signal_process(c2, method=method, parameters=parameters)
    if numpy:
        left_spec = left_spec.numpy()
        right_spec = right_spec.numpy()
    return left_spec, right_spec

def normalize(data):
    left = data[:1]
    right = data[1:]
    left = (left - torch.mean(left)) / torch.std(left)
    right = (right - torch.mean(right)) / torch.std(right)
    return torch.cat([left, right], dim=0)

if __name__ == '__main__':
    y, sr = librosa.load('sample.wav' , mono=False)
    y = torch.Tensor(y).unsqueeze(0)
    log_mel_param_dict = {'n_fft' : 2048,
                        'win_length' : 2048,
                        'hop_length' : 512,
                        'normalized' : False,
                        'sample_rate' : 48000,
                        'f_min' : 0,
                        'f_max' : None,
                        'n_mels' : 1024,
                        'power' : 1
                        }
    log_mel_param_dict['sr'] = sr
    print('raw_audio', signal_process(y, method='raw_audio', parameters=log_mel_param_dict).shape)
    print('spectrogram', signal_process(y, method='spectrogram', parameters=log_mel_param_dict).shape)
    print('logspectrogram', signal_process(y, method='logspectrogram', parameters=log_mel_param_dict).shape)
    print('melspectrogram', signal_process(y, method='melspectrogram', parameters=log_mel_param_dict).shape)
    print('logmelspectrogram', signal_process(y, method='logmelspectrogram', parameters=log_mel_param_dict).shape)
    print('mfcc', signal_process(y, method='mfcc', parameters=log_mel_param_dict).shape)
    # print('rainbowgram', signal_process(y, method='rainbowgram', parameters=log_mel_param_dict).shape)