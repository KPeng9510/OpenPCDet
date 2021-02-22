from .detector3d_template import Detector3DTemplate
from .unet.unet import UNet
from .segmentation_head import FCNMaskHead
import sys
import torch.nn as nn

from .erfnet import Net
import os
import torch.nn.functional as F
import torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from PIL import Image
import numpy as np
def one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size()[0], num_classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1) 
    return target
def one_hot_1d(data, num_classes):
     n_values = num_classes
     n_values = torch.eye(n_values)[data]
     return n_values

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        #self.segmentation_head = FCNMaskHead()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.MAP_TO_BEV.NUM_BEV_FEATURES

        self.semantic_class = 18 # change it to the version you want
        self.segmentation_head = UNet(self.num_bev_features,self.semantic_class)
        self.att = nn.Sequential(
                   nn.Conv2d(self.semantic_class,1,kernel_size=3,stride=1,padding=1,bias=False),
                   nn.Sigmoid()
                   )
    def forward(self, batch_dict):
        module_index = 0
        h,w=512, 512
        
        for cur_module in self.module_list:
            module_index += 1
            batch_dict = cur_module(batch_dict)
            
            if module_index == 2:
                """
                  encode bbox
                """
                
                points_mean = batch_dict["points_coor"]
                gt_boxes = batch_dict["gt_boxes"]
                batch_size = batch_dict["batch_size"]
                #batch = points_mean.size()
                dict_seg = []
                dict_cls_num = []
                label_b = batch_dict["dense_point"].view(batch_size,1,h,w)
                batch=label_b.size()[0]
                """
                   bounding boxes projection
                """
                for i in range(gt_boxes.size()[0]):
                    
                    points = points_mean
                    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points.unsqueeze(dim=0).float().cuda(),
                    gt_boxes[i,:,:7].unsqueeze(dim=0).float().cuda()
                    ).long().squeeze(dim=0)
                    label = label_b[i].flatten()
                    gt_boxes_indx = gt_boxes[i,:,-1]
                    gt_boxes_indx = torch.cat([torch.Tensor([0]).cuda(),gt_boxes_indx],dim=0)
                    box_idxs_of_pts +=1
                    nonzero_mask_2 = gt_boxes_indx[box_idxs_of_pts.long()] != 0
                    label[nonzero_mask_2] = gt_boxes_indx[box_idxs_of_pts.long()][nonzero_mask_2]
                    target_cr = label
                    dict_seg.append(target_cr.unsqueeze(0))

                """
                 end
                """
                targets_crr = torch.cat(dict_seg,dim=0).view(batch_size,1,h,w)
                spatial_features = batch_dict["spatial_features"]
                
                pred = self.segmentation_head(spatial_features)
                batch_dict["spatial_features"]=self.att(pred).repeat(1,self.num_bev_features,1,1)*spatial_features
                target = targets_crr.contiguous().view(batch_size,1,h,w)
                """
                    dense semantic segmentation supervision

                """
                nozero_mask = target != 0
                
                target = one_hot_1d((target[nozero_mask]-1).long(), self.semantic_class).unsqueeze(0).permute(0,2,1).cuda()
                pred = pred.permute(0,2,3,1).unsqueeze(1)[nozero_mask].squeeze().unsqueeze(0).permute(0,2,1)
                loss_seg = F.binary_cross_entropy_with_logits(pred,target,reduction='mean')
        """
           loss encoding
        """
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss + 1.2*loss_seg
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
