# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class BDLoss(nn.Module):
    """Boundary loss.

    This function is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L122>`_.  # noqa
    Licensed under the MIT License.


    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_boundary'):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    # def forward(self, bd_pre: Tensor, bd_gt: Tensor) -> Tensor:
    #     """Forward function.
    #     Args:
    #         bd_pre (Tensor): Predictions of the boundary head.
    #         bd_gt (Tensor): Ground truth of the boundary.

    #     Returns:
    #         Tensor: Loss tensor.
    #     """
    #     log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    #     target_t = bd_gt.view(1, -1).float()

    #     pos_index = (target_t == 1)
    #     neg_index = (target_t == 0)

    #     weight = torch.zeros_like(log_p)
    #     pos_num = pos_index.sum()
    #     neg_num = neg_index.sum()
    #     sum_num = pos_num + neg_num
    #     weight[pos_index] = neg_num * 1.0 / sum_num
    #     weight[neg_index] = pos_num * 1.0 / sum_num

    #     loss = F.binary_cross_entropy_with_logits(
    #         log_p, target_t, weight, reduction='mean')

    #     return self.loss_weight * loss
    
    def forward(self, seg_logits: Tensor, seg_label: Tensor) -> Tensor:
        # 首先处理mask,将255的区域标记为不参与计算  
        valid_mask = (seg_label != 255).float()  
        
        # 将logits转换为概率图  
        seg_prob = torch.sigmoid(seg_logits.squeeze(1))  # (batch, h, w)  
        
        # 创建Sobel算子  
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)  
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)  
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(seg_logits.device)  
        sobel_y = sobel_y.view(1, 1, 3, 3).to(seg_logits.device)  
        
        # 计算预测图的边界  
        prob_x = F.conv2d(seg_prob.unsqueeze(1), sobel_x, padding=1)  
        prob_y = F.conv2d(seg_prob.unsqueeze(1), sobel_y, padding=1)  
        pred_boundary = torch.sqrt(prob_x.pow(2) + prob_y.pow(2)).squeeze(1)  
        
        # 计算gt的边界  
        gt = (seg_label == 1).float()  
        gt_x = F.conv2d(gt.unsqueeze(1), sobel_x, padding=1)  
        gt_y = F.conv2d(gt.unsqueeze(1), sobel_y, padding=1)  
        gt_boundary = torch.sqrt(gt_x.pow(2) + gt_y.pow(2)).squeeze(1)  
        
        # 计算边界损失（使用Binary Cross Entropy）  
        boundary_loss = F.binary_cross_entropy(  
            pred_boundary,  
            gt_boundary,  
            reduction='none'  
        )  
        
        # 应用valid_mask  
        boundary_loss = boundary_loss * valid_mask  
        
        # 计算平均损失  
        valid_pixels = valid_mask.sum()  
        if valid_pixels > 0:  
            boundary_loss = boundary_loss.sum() / valid_pixels  
        else:  
            boundary_loss = boundary_loss.sum() * 0  
        
        return boundary_loss 
    

    @property
    def loss_name(self):
        return self.loss_name_
