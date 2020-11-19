from Model_template import Model_template
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from utils import *
from torch.utils.data import DataLoader
from make_dataset import *
from easydict import EasyDict


def get_model(model_name, hyperparameters):
    model_dict = {'Each_NN' : Each_NN(hyperparameters), 'Each_NN_diff' : Each_NN_diff(hyperparameters),
                'Angle_Each_NN' : Angle_Each_NN(hyperparameters), 
                'Class_Each_NN' : Class_Each_NN(hyperparameters),
                'Resnet' : VanillaResnet(hyperparameters, ResidualBlock),
                'My_Class_Each_NN' : My_Class_Each_NN(hyperparameters),
                'My_Resnet' : My_Resnet(hyperparameters, ResidualBlock),
                'resnet18' : resnet18(hyperparameters, ResidualBlock),
                'Angle_Each_NN' : Angle_Each_NN(hyperparameters),
                'My_Angle_Each_NN' : My_Angle_Each_NN(hyperparameters),
                'My_Angle_Resnet' : My_Angle_Resnet(hyperparameters, ResidualBlock),
                'My_Class_Resnet' : My_Class_Resnet(hyperparameters, ResidualBlock)
                }
    
    return model_dict[model_name]

def load_model_data(folder, model_name, ckpt, data_path):
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
    
    hyper.folder = folder
    model = get_model(model_name, hyper)
    model = model.load_model(ckpt)
    label_option = model.hparams.label_option
    method = model.hparams.method
    denoise = model.hparams.denoise
    print('model name :',model_name)
    print('signal process :', method)
    print('denoise :', denoise )
    
    dataset = DroneDataset(label_option=label_option, method=method, test=True, audio_dir=data_path,
                       parameters=model.hparams,denoise=denoise)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0)
    
    return model, dataloader
    
    
class Each_NN(Model_template):
    def __init__(self, hyperparameters, verbose=True):
        super().__init__(hyperparameters)
        self.verbose= verbose
        self.loss = All_MSE()
        self.conv1 = nn.Conv2d(2, 16, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 16, 2)
        self.conv3_1 = nn.Conv2d(8, 8, 16, 4)
        self.conv3_2 = nn.Conv2d(8, 2, 32, 4)
        self.conv4_1 = nn.Conv2d(8, 2, 16, 4)
        self.conv4_2 = nn.Conv2d(2, 2, 32, 4)
        
        self.linear_angle = nn.ModuleDict()
        for i in range(10):
            seq = nn.Sequential(nn.Linear(858, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_angle['%s'%i] = seq
            
        self.linear_class = nn.ModuleDict()
        for i in range(3):
            seq = nn.Sequential(nn.Linear(408, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_class['%s'%i] = seq
            
        for k, v in self.state_dict().items():
            if 'weight' in k:
                torch.nn.init.kaiming_uniform_(v, nonlinearity='relu')
            else:
                torch.nn.init.zeros_(v)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3_1(x))
        x2 = F.relu(self.conv3_2(x))
        x1 = F.relu(self.conv4_1(x1))
        x2 = F.relu(self.conv4_2(x2))

        x1 = nn.Flatten(1, -1)(x1)
        x2 = nn.Flatten(1, -1)(x2)

        output_angle = {}
        for i in range(10):
            output_angle['x_%s'%i] = self.linear_angle['%s'%i](x1)
            
        output_class = {}  
        for i in range(3):
            output_class['x_%s'%i] = self.linear_class['%s'%i](x2)
            
        x1 = torch.cat([v.view(-1, 1) for k, v in output_angle.items()], dim=1)
        x2 = torch.cat([v.view(-1, 1) for k, v in output_class.items()], dim=1)
        if (not self.training) and self.verbose:
            print(x1[:4])
            print(x2[:4])
        return (x1, x2)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
    
class Each_NN_diff(Model_template):
    def __init__(self, hyperparameters, verbose=True):
        super().__init__(hyperparameters)
        self.verbose = verbose
        self.loss = All_MSE()
        self.conv1 = nn.Conv2d(2, 16, 16, 2, )
        self.conv2 = nn.Conv2d(16, 8, 16, 2)
        self.conv3_1 = nn.Conv2d(8, 8, 16, 4)
        self.conv3_2 = nn.Conv2d(8, 2, 32, 4)
        self.conv4_1 = nn.Conv2d(8, 2, 16, 4)
        self.conv4_2 = nn.Conv2d(2, 2, 32, 4)
        
        self.linear_angle = nn.ModuleDict()
        for i in range(10):
            seq = nn.Sequential(nn.Linear(858, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_angle['%s'%i] = seq
            
        self.linear_class = nn.ModuleDict()
        for i in range(3):
            seq = nn.Sequential(nn.Linear(408, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_class['%s'%i] = seq
            
        for k, v in self.state_dict().items():
            if 'weight' in k:
                torch.nn.init.kaiming_uniform_(v, nonlinearity='relu')
            else:
                torch.nn.init.zeros_(v)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3_1(x))
        x2 = F.relu(self.conv3_2(x))
        x1 = F.relu(self.conv4_1(x1))
        x2 = F.relu(self.conv4_2(x2))

        x1 = nn.Flatten(1, -1)(x1)
        x2 = nn.Flatten(1, -1)(x2)

        output_angle = {}
        for i in range(10):
            output_angle['x_%s'%i] = self.linear_angle['%s'%i](x1)
            
        output_class = {}  
        for i in range(3):
            output_class['x_%s'%i] = self.linear_class['%s'%i](x2)
            
        x1 = torch.cat([v.view(-1, 1) for k, v in output_angle.items()], dim=1)
        x2 = torch.cat([v.view(-1, 1) for k, v in output_class.items()], dim=1)
        if (not self.training) and self.verbose:
            print(x1[:4])
            print(x2[:4])
        return (x1, x2)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
class Angle_Each_NN(Model_template):
    def __init__(self, hyperparameters, verbose=True):
        super().__init__(hyperparameters)
        self.verbose = verbose
        self.loss = nn.MSELoss()
        self.conv1 = nn.Conv2d(2, 16, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 16, 2)
        self.conv3_1 = nn.Conv2d(8, 8, 16, 4)
        self.conv4_1 = nn.Conv2d(8, 2, 16, 4)
        
        self.linear_angle = nn.ModuleDict()
        for i in range(10):
            seq = nn.Sequential(nn.Linear(858, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_angle['%s'%i] = seq

            
        for k, v in self.state_dict().items():
            if 'weight' in k:
                torch.nn.init.kaiming_uniform_(v, nonlinearity='relu')
            else:
                torch.nn.init.zeros_(v)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3_1(x))
        x1 = F.relu(self.conv4_1(x1))

        x1 = nn.Flatten(1, -1)(x1)

        output_angle = {}
        for i in range(10):
            output_angle['x_%s'%i] = self.linear_angle['%s'%i](x1)

            
        x1 = torch.cat([v.view(-1, 1) for k, v in output_angle.items()], dim=1)
        if (not self.training) and self.verbose:
            print(x1[:4])
        return x1
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
class Class_Each_NN(Model_template):
    def __init__(self, hyperparameters, verbose=True):
        super().__init__(hyperparameters)
        self.verbose = verbose
        self.loss = nn.MSELoss()
        self.conv1 = nn.Conv2d(2, 16, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 16, 2)
        self.conv3_1 = nn.Conv2d(8, 8, 16, 4)
        self.conv4_1 = nn.Conv2d(8, 2, 16, 4)
        
        self.linear_class = nn.ModuleDict()
        for i in range(3):
            seq = nn.Sequential(nn.Linear(858, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_class['%s'%i] = seq

            
        for k, v in self.state_dict().items():
            if 'weight' in k:
                torch.nn.init.kaiming_uniform_(v, nonlinearity='relu')
            else:
                torch.nn.init.zeros_(v)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3_1(x))
        x1 = F.relu(self.conv4_1(x1))

        x1 = nn.Flatten(1, -1)(x1)

        output_angle = {}
        for i in range(3):
            output_angle['x_%s'%i] = self.linear_class['%s'%i](x1)

            
        x1 = torch.cat([v.view(-1, 1) for k, v in output_angle.items()], dim=1)
        if (not self.training) and self.verbose:
            print(x1[:4])
        return x1
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]   

    
    
    
# keep dimensionality
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# reduce dimensionality
def conv1x1(in_channels, out_channels, stride=1): 
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class ResidualBlock(Model_template):
    expansion = 1
    def __init__(self, hyperparameters, in_channels, out_channels, stride=1, downsample=None):
        super().__init__(hyperparameters)
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # layer 바뀌면서 filter 개수 바뀔때 필요
        self.stride = stride

    def forward(self, x):
        identity = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # layer 바뀔때마다 filter 개수가 바뀌어서 1x1conv로 channel 맞춰주기
        if self.downsample is not None:
            identity = self.downsample(x) 
            
        out += identity
        return self.relu(out)


class VanillaResnet(Model_template):
    def __init__(self, hyperparameters, block=ResidualBlock, layers=[2, 2, 2, 2], verbose=True, 
                 hidden_nodes=[16, 32, 32, 64], num_classes=10, zero_init_residual=False):
        super(VanillaResnet, self).__init__(hyperparameters)
        self.verbose = verbose
        self.hyperparameters = hyperparameters
        self.loss = nn.MSELoss()
        #self.loss = D_angle
        self.in_channels = hidden_nodes[0]
        self.conv = nn.Conv2d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # First 7x7 conv
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer 정의 내린대로 list append 방식으로 쌓기
        for i_layer in range(len(layers)):
            if i_layer == 0:
                s1 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer])
            
            else:
                s2 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer], 2)
                s1 = nn.Sequential(s1, s2)
        self.convLayer = s1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # (batch, rest, 1, 1)
        self.fc = nn.Linear(hidden_nodes[-1] * block.expansion, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # layer 합쳐주기
    def make_layer(self, block, out_channels, blocks, stride=1):
        # filter 수 바뀔때 필요
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # layer에 residual block 쌓기
        layers = []
        layers.append(block(self.hyperparameters, self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.hyperparameters, self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input shape = (batch, channel, n_mels, time)
        out = self.conv(x) # (batch, in_channels, n_mels/2, time/2)
        out = self.bn(out) 
        out = self.relu(out)
        out = self.maxpool(out) # (batch, in_channels, n_mels/4, time/4)      
        
        out = self.convLayer(out) # (batch, hidden_nodes[-1], ?, ?)
        out = self.adaptive_pool(out) # (batch, hidden_nodes[-1], 1, 1)
        out = torch.flatten(out, 1) # (batch, hidden_nodes[-1])
        out = self.relu(self.fc(out)) # (batch x 24)
        if (not self.training) and self.verbose:
            print(out[:4])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
    
    
    
class My_Class_Each_NN(Model_template):
    def __init__(self, hyperparameters, verbose=True):
        super().__init__(hyperparameters)
        self.verbose = verbose
        self.loss = My_MSE()
        self.conv1 = nn.Conv2d(2, 16, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 16, 2)
        self.conv3_1 = nn.Conv2d(8, 8, 16, 4)
        self.conv4_1 = nn.Conv2d(8, 2, 16, 4)
        
        self.linear_class = nn.ModuleDict()
        for i in range(3):
            seq = nn.Sequential(nn.Linear(858, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_class['%s'%i] = seq

            
        for k, v in self.state_dict().items():
            if 'weight' in k:
                torch.nn.init.kaiming_uniform_(v, nonlinearity='relu')
            else:
                torch.nn.init.zeros_(v)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3_1(x))
        x1 = F.relu(self.conv4_1(x1))

        x1 = nn.Flatten(1, -1)(x1)

        output_angle = {}
        for i in range(3):
            output_angle['x_%s'%i] = self.linear_class['%s'%i](x1)

            
        x1 = torch.cat([v.view(-1, 1) for k, v in output_angle.items()], dim=1)
        if (not self.training) and self.verbose:
            print(x1[:4])
        return x1
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]   



class My_Resnet(Model_template):
    def __init__(self, hyperparameters, block=ResidualBlock, layers=[2, 2, 2, 2], verbose=True, 
                 hidden_nodes=[16, 32, 32, 64], num_classes=10, zero_init_residual=False):
        super(My_Resnet, self).__init__(hyperparameters)
        self.hyperparameters = hyperparameters
        self.verbose = verbose
        self.loss = My_MSE()
        #self.loss = D_angle
        self.in_channels = hidden_nodes[0]
        self.conv = nn.Conv2d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # First 7x7 conv
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer 정의 내린대로 list append 방식으로 쌓기
        for i_layer in range(len(layers)):
            if i_layer == 0:
                s1 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer])
            
            else:
                s2 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer], 2)
                s1 = nn.Sequential(s1, s2)
        self.convLayer = s1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # (batch, rest, 1, 1)
        self.fc = nn.Linear(hidden_nodes[-1] * block.expansion, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # layer 합쳐주기
    def make_layer(self, block, out_channels, blocks, stride=1):
        # filter 수 바뀔때 필요
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # layer에 residual block 쌓기
        layers = []
        layers.append(block(self.hyperparameters, self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.hyperparameters, self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input shape = (batch, channel, n_mels, time)
        out = self.conv(x) # (batch, in_channels, n_mels/2, time/2)
        out = self.bn(out) 
        out = self.relu(out)
        out = self.maxpool(out) # (batch, in_channels, n_mels/4, time/4)      
        
        out = self.convLayer(out) # (batch, hidden_nodes[-1], ?, ?)
        out = self.adaptive_pool(out) # (batch, hidden_nodes[-1], 1, 1)
        out = torch.flatten(out, 1) # (batch, hidden_nodes[-1])
        out = self.relu(self.fc(out)) # (batch x 24)
        if (not self.training) and self.verbose:
            print(out[:4])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
    
class resnet18(Model_template):
    def __init__(self, hyperparameters, block=ResidualBlock, layers=[2, 2, 2, 2], verbose=True, 
                 hidden_nodes=[32, 64, 128, 256], num_classes=10, zero_init_residual=False):
        super(resnet18, self).__init__(hyperparameters)
        self.verbose = verbose
        self.loss = nn.MSELoss()
        self.hyperparameters = hyperparameters
        #self.loss = D_angle
        self.in_channels = hidden_nodes[0]
        self.conv = nn.Conv2d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # First 7x7 conv
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer 정의 내린대로 list append 방식으로 쌓기
        for i_layer in range(len(layers)):
            if i_layer == 0:
                s1 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer])
            
            else:
                s2 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer], 2)
                s1 = nn.Sequential(s1, s2)
        self.convLayer = s1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # (batch, rest, 1, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_nodes[-1] * block.expansion, hidden_nodes[-1] * block.expansion), nn.Linear(hidden_nodes[-1] * block.expansion, num_classes))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # layer 합쳐주기
    def make_layer(self, block, out_channels, blocks, stride=1):
        # filter 수 바뀔때 필요
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # layer에 residual block 쌓기
        layers = []
        layers.append(block(self.hyperparameters, self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.hyperparameters, self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input shape = (batch, channel, n_mels, time)
        out = self.conv(x) # (batch, in_channels, n_mels/2, time/2)
        out = self.bn(out) 
        out = self.relu(out)
        out = self.maxpool(out) # (batch, in_channels, n_mels/4, time/4)      
        
        out = self.convLayer(out) # (batch, hidden_nodes[-1], ?, ?)
        out = self.adaptive_pool(out) # (batch, hidden_nodes[-1], 1, 1)
        out = torch.flatten(out, 1) # (batch, hidden_nodes[-1])
        out = self.fc(out) # (batch x 24)
        if (not self.training) and self.verbose:
            print(f'out: {out}')
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
class Angle_Each_NN(Model_template):
    def __init__(self, hyperparameters, verbose=True):
        super().__init__(hyperparameters)
        self.verbose = verbose
        self.loss = nn.MSELoss()
        self.conv1 = nn.Conv2d(2, 16, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 16, 2)
        self.conv3_1 = nn.Conv2d(8, 8, 16, 4)
        self.conv4_1 = nn.Conv2d(8, 2, 16, 4)
        
        self.linear_class = nn.ModuleDict()
        for i in range(10):
            seq = nn.Sequential(nn.Linear(858, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_class['%s'%i] = seq

            
        for k, v in self.state_dict().items():
            if 'weight' in k:
                torch.nn.init.kaiming_uniform_(v, nonlinearity='relu')
            else:
                torch.nn.init.zeros_(v)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3_1(x))
        x1 = F.relu(self.conv4_1(x1))

        x1 = nn.Flatten(1, -1)(x1)

        output_angle = {}
        for i in range(10):
            output_angle['x_%s'%i] = self.linear_class['%s'%i](x1)

            
        x1 = torch.cat([v.view(-1, 1) for k, v in output_angle.items()], dim=1)
        if (not self.training) and self.verbose:
            print(x1[:4])
        return x1
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
class My_Angle_Each_NN(Model_template):
    def __init__(self, hyperparameters, verbose=True):
        super().__init__(hyperparameters)
        self.verbose = verbose
        self.loss = My_MSE_A()
        self.conv1 = nn.Conv2d(2, 16, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 16, 2)
        self.conv3_1 = nn.Conv2d(8, 8, 16, 4)
        self.conv4_1 = nn.Conv2d(8, 2, 16, 4)
        
        self.linear_class = nn.ModuleDict()
        for i in range(10):
            seq = nn.Sequential(nn.Linear(858, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 1),
                                nn.ReLU(inplace=True)
                               )
            self.linear_class['%s'%i] = seq

            
        for k, v in self.state_dict().items():
            if 'weight' in k:
                torch.nn.init.kaiming_uniform_(v, nonlinearity='relu')
            else:
                torch.nn.init.zeros_(v)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3_1(x))
        x1 = F.relu(self.conv4_1(x1))

        x1 = nn.Flatten(1, -1)(x1)

        output_angle = {}
        for i in range(10):
            output_angle['x_%s'%i] = self.linear_class['%s'%i](x1)

            
        x1 = torch.cat([v.view(-1, 1) for k, v in output_angle.items()], dim=1)
        if (not self.training) and self.verbose:
            print(x1[:4])
        return x1
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]   
    
    
    
class My_Angle_Resnet(Model_template):
    def __init__(self, hyperparameters, block=ResidualBlock, layers=[2, 2, 2, 2], verbose=True, 
                 hidden_nodes=[16, 32, 32, 64], num_classes=10, zero_init_residual=False):
        super(My_Angle_Resnet, self).__init__(hyperparameters)
        self.verbose = verbose
        self.hyperparameters = hyperparameters
        self.loss = My_MSE_A()
        #self.loss = D_angle
        self.in_channels = hidden_nodes[0]
        self.conv = nn.Conv2d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # First 7x7 conv
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer 정의 내린대로 list append 방식으로 쌓기
        for i_layer in range(len(layers)):
            if i_layer == 0:
                s1 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer])
            
            else:
                s2 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer], 2)
                s1 = nn.Sequential(s1, s2)
        self.convLayer = s1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # (batch, rest, 1, 1)
        self.fc = nn.Linear(hidden_nodes[-1] * block.expansion, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # layer 합쳐주기
    def make_layer(self, block, out_channels, blocks, stride=1):
        # filter 수 바뀔때 필요
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # layer에 residual block 쌓기
        layers = []
        layers.append(block(self.hyperparameters, self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.hyperparameters, self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input shape = (batch, channel, n_mels, time)
        out = self.conv(x) # (batch, in_channels, n_mels/2, time/2)
        out = self.bn(out) 
        out = self.relu(out)
        out = self.maxpool(out) # (batch, in_channels, n_mels/4, time/4)      
        
        out = self.convLayer(out) # (batch, hidden_nodes[-1], ?, ?)
        out = self.adaptive_pool(out) # (batch, hidden_nodes[-1], 1, 1)
        out = torch.flatten(out, 1) # (batch, hidden_nodes[-1])
        out = self.relu(self.fc(out)) # (batch x 24)
        if (not self.training) and self.verbose:
            print(out[:4])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
    
class My_Class_Resnet(Model_template):
    def __init__(self, hyperparameters, block=ResidualBlock, layers=[2, 2, 2, 2], verbose=True, 
                 hidden_nodes=[16, 32, 32, 64], num_classes=10, zero_init_residual=False):
        super(My_Class_Resnet, self).__init__(hyperparameters)
        self.hyperparameters = hyperparameters
        self.verbose = verbose
        self.loss = My_MSE_small()
        #self.loss = D_angle
        self.in_channels = hidden_nodes[0]
        self.conv = nn.Conv2d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # First 7x7 conv
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer 정의 내린대로 list append 방식으로 쌓기
        for i_layer in range(len(layers)):
            if i_layer == 0:
                s1 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer])
            
            else:
                s2 = self.make_layer(block, hidden_nodes[i_layer], layers[i_layer], 2)
                s1 = nn.Sequential(s1, s2)
        self.convLayer = s1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # (batch, rest, 1, 1)
        self.fc = nn.Linear(hidden_nodes[-1] * block.expansion, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # layer 합쳐주기
    def make_layer(self, block, out_channels, blocks, stride=1):
        # filter 수 바뀔때 필요
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # layer에 residual block 쌓기
        layers = []
        layers.append(block(self.hyperparameters, self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.hyperparameters, self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input shape = (batch, channel, n_mels, time)
        out = self.conv(x) # (batch, in_channels, n_mels/2, time/2)
        out = self.bn(out) 
        out = self.relu(out)
        out = self.maxpool(out) # (batch, in_channels, n_mels/4, time/4)      
        
        out = self.convLayer(out) # (batch, hidden_nodes[-1], ?, ?)
        out = self.adaptive_pool(out) # (batch, hidden_nodes[-1], 1, 1)
        out = torch.flatten(out, 1) # (batch, hidden_nodes[-1])
        out = self.relu(self.fc(out)) # (batch x 24)
        if (not self.training) and self.verbose:
            print(out[:4])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]