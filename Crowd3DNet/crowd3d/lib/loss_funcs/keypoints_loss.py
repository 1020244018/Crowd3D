from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import pickle
import numpy as np
from evaluation import compute_error_verts, compute_similarity_transform, compute_similarity_transform_torch, \
                    batch_compute_similarity_transform_torch, compute_mpjpe

def compute_side_ground_loss(dist_ori):
    '''
    dist: m, 6890
    when dist>0, loss=0
    when dist<0, loss=max(dist<0)
    '''
    dist=dist_ori.clone()
    neg_inds = dist.lt(0).float()
    dist_loss=torch.abs(dist)*neg_inds # m, 6890
    dist_keep_loss=torch.max(dist_loss, axis=-1)[0] # m,
    side_ground_loss=dist_keep_loss.mean(-1)
    # print('dist', dist.shape, 'dist_loss', dist_loss.shape, 'dist_keep_loss', dist_keep_loss.shape, 'side_ground_loss', side_ground_loss.shape)
    return side_ground_loss

def batch_kp_2d_l2_loss(real, pred, weights=None):

    vis = (real>-2).sum(-1)==real.shape[-1] #-1 -> -2

    pred[~vis] = real[~vis]

    error = torch.norm(real-pred, p=2, dim=-1)
    if weights is not None:
        error = error * weights.to(error.device)
    loss = error.sum(-1) / (1e-6+vis.sum(-1))
    return loss

def batch_bbox_2d_l2_loss(real, pred, weights=None):

    error = torch.norm(real-pred, p=2, dim=-1)
    if weights is not None:
        error = error * weights.to(error.device)
    loss = error.sum(-1) / error.shape[-1]
    return loss

def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)

def calc_mpjpe(real, pred, align_inds=None, sample_wise=True, trans=None, return_org=False):
    vis_mask = real[:,:,0] != -2.
    if align_inds is not None:
        pred_aligned = align_by_parts(pred,align_inds=align_inds)
        if trans is not None:
            pred_aligned += trans
        real_aligned = align_by_parts(real,align_inds=align_inds)
    else:
        pred_aligned, real_aligned = pred, real
    mpjpe_each = compute_mpjpe(pred_aligned, real_aligned, vis_mask, sample_wise=sample_wise)
    if return_org:
        return mpjpe_each, (real_aligned, pred_aligned, vis_mask)
    return mpjpe_each

def calc_pampjpe(real, pred, sample_wise=True,return_transform_mat=False):
    real, pred = real.float(), pred.float()
    # extracting the keypoints that all samples have the annotations
    vis_mask = (real[:,:,0] != -2.).sum(0)==len(real)
    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(pred[:,vis_mask], real[:,vis_mask])
    pa_mpjpe_each = compute_mpjpe(pred_tranformed, real[:,vis_mask], sample_wise=sample_wise)
    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each