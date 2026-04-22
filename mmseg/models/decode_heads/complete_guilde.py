# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ..losses import accuracy
from ..utils import resize
from mmseg.utils import SampleList


@MODELS.register_module()
class CompleteGuideHead(BaseDecodeHead):
    """the segmentation head.

    Args:
        num_classes (int): the classes num.
        in_channels (int): the input channels.
        use_dw (bool): if to use deepwith convolution.
        dropout_ratio (float): Probability of an element to be zeroed.
            Default 0.0。
        align_corners (bool, optional): Geometrically, we consider the pixels
            of the input and output as squares rather than points.
        upsample (str): the upsample method.
        out_channels (int): the output channel.
        conv_cfg (dict): Config dict for convolution layer.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 use_dw=True,
                 dropout_ratio=0.1,
                 align_corners=False,
                 upsample='intepolate',
                 out_channels=None,
                 threshold=0.5,
                 conv_cfg=dict(type='Conv'),
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),):

        super().__init__(in_channels=in_channels, channels=in_channels,
                         num_classes=num_classes, loss_decode=loss_decode,
                         out_channels=out_channels, threshold=threshold)
        self.align_corners = align_corners
        self.last_channels = in_channels
        self.upsample = upsample
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.linear_fuse = ConvModule(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            bias=False,
            groups=self.last_channels if use_dw else 1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio)
        # self.conv_seg = build_conv_layer(
        #     conv_cfg, self.last_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        # x, x_hw = x[0], x[1]
        x_hw = x.shape[2:]
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        if self.upsample == 'intepolate' or self.training or \
                self.num_classes < 30:
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)
        elif self.upsample == 'vim':
            labelset = torch.unique(torch.argmax(x, 1))
            x = torch.gather(x, 1, labelset)
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)

            pred = torch.argmax(x, 1)
            pred_retrieve = torch.zeros(pred.shape, dtype=torch.int32)
            for i, val in enumerate(labelset):
                pred_retrieve[pred == i] = labelset[i].cast('int32')

            x = pred_retrieve
        else:
            raise NotImplementedError(self.upsample, ' is not implemented')

        return x

    # def predict(self, inputs, batch_img_metas: List[dict], test_cfg,
    #             **kwargs) -> List[Tensor]:
    #     """Forward function for testing, only ``pam_cam`` is used."""
    #     seg_logits = self.forward(inputs)[0]
    #     return seg_logits
    
    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        seg_label[seg_label == 2] = 1

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        # print(seg_logits.shape, seg_label.shape)
        return loss
