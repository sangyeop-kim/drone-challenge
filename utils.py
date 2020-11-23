import torch
from torch.nn import _reduction
from torch.nn import Module
from torch import Tensor
from easydict import EasyDict
import json
import numpy as np
from torch.nn import functional as F


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
            
            
class D_loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', 
                device_num: int = -1) -> None:
        super(D_loss, self).__init__(size_average, reduce, reduction)
        self.device_num = device_num
        
    def forward(self, target: Tensor,  pred: Tensor) -> Tensor:
        target_angle = target[0]
        target_class = target[1]
        pred_angle = pred[0]
        pred_class = pred[1]
        
        D_angle = score_angle(target_angle, pred_angle, self.device_num)
        D_class = score_class(target_class, pred_class)

        D_total = D_angle * 0.8 + D_class * 0.2
        return D_total


class All_MSE(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(All_MSE, self).__init__(size_average, reduce, reduction)
        
    def forward(self, target: Tensor,  pred: Tensor) -> Tensor:
        target_angle = target[0]
        target_class = target[1]
        pred_angle = pred[0]
        pred_class = pred[1]

        D_total = F.mse_loss(pred_angle, target_angle) * 1.1 + F.mse_loss(pred_class, target_class) * 0.2
        return D_total

def score_angle(target, pred, device_num=-1):
    if device_num == -1:
        device = 'cpu'
    else:
        device = 'cuda:%s' % device_num
        
    zero = torch.zeros(target.shape[0], 2).to(device)
    target = torch.cat((zero, target, zero), dim=1)
    pred = torch.cat((zero, pred, zero), dim=1)
    angle_distance = 0

    for num in range(target.shape[1]-4):
        weight = torch.tensor([0.05, 0.1, 0.7, 0.1, 0.05]).to(device)
        wma_target = torch.sum(target[:,num:num+5] * weight, dim=1)
        wma_pred = torch.sum(pred[:,num:num+5] * weight, dim=1)
        angle_distance += torch.sum((wma_target - wma_pred)**2)
        
    return angle_distance


def score_class(target, pred):
    class_distance = 0
    
    for num in range(3):
        class_distance += torch.sum((target[:,num] - pred[:,num])**2)
        
    return class_distance


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class All_MSE_diff(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(All_MSE, self).__init__(size_average, reduce, reduction)
        
    def forward(self, target: Tensor,  pred: Tensor) -> Tensor:
        target_angle = target[0]
        target_class = target[1]
        pred_angle = pred[0]
        pred_class = pred[1]
        sum_angle = torch.sum(pred_angle, dim=1)
        sum_class = torch.sum(pred_class, dim=1)
        D_total = F.mse_loss(pred_angle, target_angle) * 1.0 + F.mse_loss(pred_class, target_class) * 0.2
        D_total += 0.8 * F.mse_loss(sum_angle, sum_class)
        return D_total
    
    
class My_MSE(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(My_MSE, self).__init__(size_average, reduce, reduction)
        
    def forward(self, target: Tensor,  pred: Tensor) -> Tensor:
        sum_target = torch.sum(target, dim=1)
        sum_pred = torch.sum(pred, dim=1)
        
        D_total = F.mse_loss(pred, target) * 1. + F.mse_loss(sum_target, sum_pred) * 0.2
        return D_total
    
class My_MSE_A(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(My_MSE_A, self).__init__(size_average, reduce, reduction)
        
    def forward(self, target: Tensor,  pred: Tensor) -> Tensor:
        sum_target = torch.sum(target, dim=1)
        sum_pred = torch.sum(pred, dim=1)
        
        D_total = F.mse_loss(pred, target) * 1. + F.mse_loss(sum_target, sum_pred) * 0.07
        return D_total
    
class My_MSE_small(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(My_MSE_small, self).__init__(size_average, reduce, reduction)
        
    def forward(self, target: Tensor,  pred: Tensor) -> Tensor:
        sum_target = torch.sum(target, dim=1)
        sum_pred = torch.sum(pred, dim=1)
        
        D_total = F.mse_loss(pred, target) * 1. + F.mse_loss(sum_target, sum_pred) * 0.1
        return D_total
    
    
def inference_rule(ang, cls_=None):
    if cls_ is not None:
        if np.max(ang) < 1.:
            ang = ang == np.max(ang)
        else:
            ang = ang

        ang = list(ang.astype(np.int64))
        total_ang = np.sum(ang)

        ori_cls_ = cls_
        if np.max(cls_) < 1.:
            cls_ = cls_ == np.max(cls_)
        else:
            cls_ = cls_

        cls_ = list(cls_.astype(np.int64))
        total_cls_ = np.sum(cls_)
        diff = total_cls_ - total_ang 
        if diff > 0:
            count = 0
            non_zero_rank = np.argsort(ori_cls_)[::-1][ori_cls_[np.argsort(ori_cls_)[::-1]]>0]
            sub = [0, 0, 0]
            for i in range(diff):
                if count == len(non_zero_rank):
                    count=0
                    
                if sub[non_zero_rank[count]] == cls_[non_zero_rank[count]]:
                    count = 0
                    sub[non_zero_rank[count]] += 1
                else:
                    sub[non_zero_rank[count]] += 1
                    count += 1
            cls_ -= np.array(sub)
        return ang, cls_
    else:
        if np.max(ang) < 1.:
            ang = ang == np.max(ang)
        else:
            ang = ang

        return list(ang.astype(np.int64))
    