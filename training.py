import os
from make_dataset import *
from easydict import EasyDict
from make_dataset import DroneDataset
from models import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--auto_lr_find', type=bool, default=False)
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--folder', type=str, default='../best_model')
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--win_length', type=int, default=2048)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--normalized', type=bool, default=True)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--power', type=int, default=1)
    parser.add_argument('--n_mels', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--label_option', type=str, default='all', choices=['all', 'angle', 'class'])
    parser.add_argument('--model', type=str, default='Each_NN')
    parser.add_argument('--datatype', type=str, default='logmel', choices=['logmel', 'reduce'])
    parser.add_argument('--denoise', type=bool, default=False)
    parser.add_argument('--n_std_thresh', type=float, default=2.1)
    parser.add_argument('--method', type=str, default='logmelspectrogram', choices=['raw_audio', 'spectrogram', 
                                                                            'logspectrogram', 
                                                                            'melspectrogram', 
                                                                            'logmelspectrogram', 'mfcc', 
                                                                            'stft', 'librosa_cqt', 
                                                                            'librosa_chroma_cqt', 
                                                                            'librosa_chroma_cens', 
                                                                            'librosa_rms', 'rainbowgram'])
    
    args = parser.parse_args()
    
    lr = args.lr
    max_epochs = args.max_epochs
    step_size = args.step_size
    gamma = args.gamma
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    gpus = args.gpus
    num_workers = args.num_workers
    auto_lr_find = args.auto_lr_find
    save_top_k = args.save_top_k
    folder = args.folder
    early_stopping = args.early_stopping
    patience = args.patience
    label_option = args.label_option
    model = args.model
    method = args.method
    win_length = args.win_length
    hop_length = args.hop_length
    normalized = args.normalized
    sample_rate = args.sample_rate
    n_mels = args.n_mels
    power = args.power
    denoise = args.denoise
    ckpt = args.ckpt
    n_std_thresh = args.n_std_thresh
    datatype = args.datatype

    
    hyperparameters = EasyDict({'lr' : lr,
                                'max_epochs' :max_epochs,
                                'step_size' : step_size,
                                'gamma' : gamma,
                                'batch_size' : batch_size,
                                'test_batch_size' : test_batch_size,
                                'gpus' : [gpus],
                                'num_workers' : num_workers,
                                'auto_lr_find' : auto_lr_find,
                                'save_top_k' : save_top_k,
                                'folder' : folder,
                                'early_stopping' : early_stopping,
                                'patience' : patience,
                                'n_fft' : win_length,
                                'win_length' : win_length,
                                'hop_length' : hop_length,
                                'normalized' : normalized,
                                'sample_rate' : sample_rate,
                                'f_min' : 0,
                                'f_max' : None,
                                'n_mels' : n_mels,
                                'power' : power,
                                'model' : model,
                                'method' : method,
                                'label_option' : label_option,
                                'denoise' : denoise,
                                'n_std_thresh' : n_std_thresh,
                                'datatype' : datatype                 
                                })

    if not os.path.isdir(hyperparameters['folder']) :
        os.mkdir(hyperparameters['folder'])
        
    model = get_model(model, hyperparameters)
    
    dataset_train = DroneDataset(label_option=label_option, method=method, parameters=hyperparameters,
                           denoise=denoise, validation=False)
    dataset_validation = DroneDataset(label_option=label_option, method=method, parameters=hyperparameters,
                           denoise=denoise, validation=True)

    print(dataset_train[0][0].shape)
    if ckpt is not None:
        model = model.load_model(ckpt)
    
    model.fit(dataset_train, dataset_validation)    
    
    
if __name__=="__main__":
	main()