# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
from mmseg.registry import MODELS
# 这两个会报ModuleNotFoundError，需要install对应的库
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

@MODELS.register_module()
class ShapeLoss(nn.Module):

    def __init__(self,
                 ):
        '''
        init部分...
        '''

    # 计算gt的符号距离图（SDM）
    def compute_sdf(img_gt, out_shape):
        """
        img_gt: 就是groundtruth (batch_size , H , W)
        out_shape: img_gt的形状: batch_size * H * W
        """

        img_gt = img_gt.astype(np.uint8)
        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch_size
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b] = sdf
                
        return normalized_sdf
    
    # 计算dice_loss
    def dice_loss(score, target):
        '''
        score: 经过sigmond放缩后的predict (batch_size , H , W)
        target: 就是groundtruth (batch_size , H , W)
        '''

        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self,
                pt,
                gt,
                ):
        """Forward function.

        Args:
            pt (torch.Tensor): (batch_size , 1 , H , W).
            gt (torch.Tensor): (batch_size , 1 , H , W).
            两者都是4维张量

            第2维的值需要是1

        Returns:
            torch.Tensor: 形状损失
        """
        # tanh激活函数
        pt_tanh=nn.Tanh(pt) # pt_tanh就是predict的SDM
        # sigmond激活函数
        pt_soft = torch.sigmoid(pt)
        
        # gt (batch_size , 1 , H , W) -> (batch_size , H , W)
        gt = gt.squeeze(1)

        batch_size=pt.shape[0]
        # 计算SDM
        with torch.no_grad():
            gt_sdm = self.compute_sdf(gt.cpu().numpy(), pt[:batch_size, 0, ...].shape) # (batch_size , 1 , H , W) -> (batch_size , H , W)
            gt_sdm = torch.from_numpy(gt_sdm).float().cuda()

        # 计算两部分损失函数
        loss_sdm_mse = MSELoss(pt_tanh[:batch_size, 0, ...], gt_sdm) # (batch_size , 1 , H , W) -> (batch_size , H , W)
        loss_seg_dice = self.dice_loss(pt_soft[:batch_size, 0, ...], gt[:batch_size] == 1) # 这里似乎与直接写gt没什么区别

        loss=loss_seg_dice + 0.3 * loss_sdm_mse # 平衡参数默认为 0.3
        return loss
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
