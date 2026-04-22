# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ..cat_seg import clip_wrapper
from ..cat_seg.clip_templates import (IMAGENET_TEMPLATES,
                                    IMAGENET_TEMPLATES_SELECT)
import json
from mmdet.models.layers import DetrTransformerDecoder
from mmseg.utils import ConfigType, SampleList
from mmdet.models.layers import SinePositionalEncoding
from typing import List, Tuple
from ..utils import nchw_to_nlc, nlc_to_nchw
from ..losses import accuracy
from ..utils import resize


@MODELS.register_module()
class ClipGuideHead(BaseDecodeHead):
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
                 clip_pretrained: str,
                 class_json: str,
                 num_classes,
                 clip_dim,
                 in_channels,
                 use_dw=True,
                 dropout_ratio=0.1,
                 align_corners=False,
                 upsample='intepolate',
                 out_channels=None,
                 conv_cfg=dict(type='Conv'),
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 prompt_depth: int = 0,
                 prompt_length: int = 0,
                 prompt_ensemble_type: str = 'imagenet',
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_decode=[dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)],
                 **kwargs):

        super().__init__(in_channels=in_channels, channels=in_channels, num_classes=num_classes, loss_decode=loss_decode)
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
        
         # prepare clip templates
        self.prompt_ensemble_type = prompt_ensemble_type
        if self.prompt_ensemble_type == 'imagenet_select':
            prompt_templates = IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == 'imagenet':
            prompt_templates = IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == 'single':
            prompt_templates = [
                # 'A photo of a {} in the scene',
                'a cropped photo of {} in the scene'
            ]
        else:
            raise NotImplementedError
        self.prompt_templates = prompt_templates
        
        # build CLIP model
        with open(class_json) as f_in:
            self.class_texts = json.load(f_in)
        assert (self.class_texts)  is not None
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        
        # for OpenAI models
        clip_model, clip_preprocess = clip_wrapper.load(
            clip_pretrained,
            device=device,
            jit=False,
            prompt_depth=prompt_depth,
            prompt_length=prompt_length)
        
        # pre-encode classes text prompts
        self.text_features = self.class_embeddings(self.class_texts,
                                              prompt_templates, clip_model,
                                              device).float()
        
        self.down_text_feature = nn.Sequential(
            nn.Linear(clip_dim, clip_dim), nn.ReLU(inplace=True),
            nn.Linear(clip_dim, clip_dim), nn.ReLU(inplace=True),
            nn.Linear(clip_dim, in_channels))
        
        self.decoder_pe = SinePositionalEncoding(**positional_encoding)
        
        self.transformer_decoder = DetrTransformerDecoder(
            **kwargs['transformer_decoder'])
        
        self.key_pos = nn.Embedding(num_classes, in_channels)
        
        self.down_input = ConvModule(
            in_channels,
            in_channels,
            3,
            2,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.mlp_input = ConvModule(
            in_channels,
            in_channels,
            3,
            1,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
    
    @torch.no_grad()
    def class_embeddings(self,
                         classnames,
                         templates,
                         clip_model,
                         device='cpu'):
        """Convert class names to text embeddings by clip model.

        Args:
            classnames (list): loaded from json file.
            templates (dict): text template.
            clip_model (nn.Module): prepared clip model.
            device (str | torch.device): loading device of text
                encoder results.
        """
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname)
                         for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).to(device)
            else:
                texts = clip_wrapper.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(
                    len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        # # 1. 对模板维度取平均  
        # zeroshot_weights = zeroshot_weights.mean(dim=0)  # 去掉keepdim=True  
        # # 2. 再次归一化  
        # zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(dim=-1, keepdim=True)
        # print(zeroshot_weights.shape)
        return zeroshot_weights

    def forward(self, x):
        # x, x_hw = x[0], x[1]
        x = self.down_input(x)
        x = x + self.mlp_input(x)
        feat_size = x.shape[2:]
        
 
        # input_img_h, input_img_w = x_hw
        batch_size = x.shape[0]
        padding_mask = x.new_ones((batch_size, *x.shape[-2:]),
                                      dtype=torch.float32)
        pos_embed = self.decoder_pe(padding_mask)
        pos_embed = nchw_to_nlc(pos_embed)
        
        text_features = self.down_text_feature(self.text_features)
        text_features = text_features.expand(batch_size, *text_features.shape[1:])
        # print(text_features.shape, self.key_pos.weight.expand(batch_size, *(self.key_pos.weight.shape)).shape)
        
        # print(x.shape, text_features.shape)
        out_dec = self.transformer_decoder(
            query=nchw_to_nlc(x),
            key=text_features,
            value=text_features,
            query_pos=pos_embed,
            key_pos=self.key_pos.weight.expand(batch_size, *(self.key_pos.weight.shape)),
            key_padding_mask=None)
        
        # print(out_dec.shape)
        x = nlc_to_nchw(out_dec[-1], feat_size)
        # print(x.shape)
        
        x = self.linear_fuse(x)
        
        # x = self.dropout(x)
        # x = self.conv_seg(x)
        
        output = torch.einsum('bchw,blc->blhw', x, text_features)

        return output

    # def predict(self, inputs, batch_img_metas: List[dict], test_cfg,
    #             **kwargs) -> List[Tensor]:
    #     """Forward function for testing, only ``pam_cam`` is used."""
    #     seg_logits = self.forward(inputs)[0]
    #     return seg_logits
    # def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
    #          train_cfg: ConfigType) -> dict:
    #     """Forward function for training.

    #     Args:
    #         inputs (Tuple[Tensor]): List of multi-level img features.
    #         batch_data_samples (list[:obj:`SegDataSample`]): The seg
    #             data samples. It usually includes information such
    #             as `img_metas` or `gt_semantic_seg`.
    #         train_cfg (dict): The training config.

    #     Returns:
    #         dict[str, Tensor]: a dictionary of loss components
    #     """
    #     seg_logits = self.forward(inputs, batch_data_samples)
    #     losses = self.loss_by_feat(seg_logits, batch_data_samples)
    #     return losses
    
    
    # def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
    #             test_cfg: ConfigType) -> Tensor:
    #     """Forward function for prediction.

    #     Args:
    #         inputs (Tuple[Tensor]): List of multi-level img features.
    #         batch_img_metas (dict): List Image info where each dict may also
    #             contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
    #             'ori_shape', and 'pad_shape'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
    #         test_cfg (dict): The testing config.

    #     Returns:
    #         Tensor: Outputs segmentation logits map.
    #     """
    #     seg_logits = self.forward(inputs, batch_img_metas)

    #     return self.predict_by_feat(seg_logits, batch_img_metas)
    
    # def loss_by_feat(self, seg_logits: Tensor,
    #                  batch_data_samples: SampleList) -> dict:
    #     """Compute segmentation loss.

    #     Args:
    #         seg_logits (Tensor): The output from decode head forward function.
    #         batch_data_samples (List[:obj:`SegDataSample`]): The seg
    #             data samples. It usually includes information such
    #             as `metainfo` and `gt_sem_seg`.

    #     Returns:
    #         dict[str, Tensor]: a dictionary of loss components
    #     """

    #     seg_label = self._stack_batch_gt(batch_data_samples)
    #     loss = dict()
    #     seg_logits = resize(
    #         input=seg_logits,
    #         size=seg_label.shape[2:],
    #         mode='bilinear',
    #         align_corners=self.align_corners)
    #     if self.sampler is not None:
    #         seg_weight = self.sampler.sample(seg_logits, seg_label)
    #     else:
    #         seg_weight = None
    #     seg_label = seg_label.squeeze(1)

    #     if not isinstance(self.loss_decode, nn.ModuleList):
    #         losses_decode = [self.loss_decode]
    #     else:
    #         losses_decode = self.loss_decode
    #     for loss_decode in losses_decode:
    #         if loss_decode.loss_name not in loss:
    #             loss[loss_decode.loss_name] = loss_decode(
    #                 seg_logits,
    #                 seg_label,)
    #                 # weight=seg_weight,
    #                 # ignore_index=self.ignore_index)
    #         else:
    #             loss[loss_decode.loss_name] += loss_decode(
    #                 seg_logits,
    #                 seg_label,)
    #                 # weight=seg_weight,
    #                 # ignore_index=self.ignore_index)

    #     loss['acc_seg'] = accuracy(
    #         seg_logits, seg_label, ignore_index=self.ignore_index)
    #     return loss