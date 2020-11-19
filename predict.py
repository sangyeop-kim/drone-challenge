import sys
import torch
from utils import *
from models import *
import numpy as np
import json

data_path = sys.argv[1]

folder = 'model_ckpt'

###### angle #####
ckpt = 'angle/epoch=124_val_loss=0.0683'
model_name = 'My_Angle_Resnet'
print('start angle')
angle_model, angle_dataloader = load_model_data(folder, model_name, ckpt, data_path)
print()

###################
ckpt = 'class/epoch=174_val_loss=0.5850'
model_name = 'My_Resnet'
print('start class')
class_model, class_dataloader = load_model_data(folder, model_name, ckpt, data_path)

angle_model.verbose = False
angle_model = angle_model.cuda()

class_model.verbose = False
class_model = class_model.cuda()


results = []

for (angle_batch, id_), (class_batch, id_) in zip(angle_dataloader, class_dataloader):
    
    angle_batch = angle_batch.cuda()
    class_batch = class_batch.cuda()
    
    with torch.no_grad():
        angle_model.eval()
        class_model.eval()
        angle_out = angle_model(angle_batch).detach().cpu().numpy()
        class_out = class_model(class_batch).detach().cpu().numpy()
        
        for num, (ang, cls_) in enumerate(zip(angle_out, class_out)):
            id_dict = {}
            id_dict['id'] = int(id_[num].item())
            a, c = inference_rule(ang, cls_)
            id_dict['angle'] = a 
            id_dict['class'] = c
            results.append(id_dict)
        
final_output = {}
final_output['track3_results'] = results
    
    
with open('t3_res_0107.json', 'w') as f:
    json.dump(final_output, f, cls=NpEncoder)

print(final_output)
print('finish')
