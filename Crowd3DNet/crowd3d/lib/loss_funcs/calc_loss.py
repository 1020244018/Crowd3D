from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import time
import pickle
import numpy as np

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from config import args
import constants

from utils.center_utils import denormalize_center
from loss_funcs.params_loss import batch_l2_loss_param,batch_l2_loss
from loss_funcs.keypoints_loss import batch_kp_2d_l2_loss, calc_mpjpe, calc_pampjpe, batch_bbox_2d_l2_loss, compute_side_ground_loss
from loss_funcs.maps_loss import focal_loss, JointsMSELoss
from loss_funcs.prior_loss import angle_prior, MaxMixturePrior
from maps_utils.centermap import CenterMap

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.gmm_prior = MaxMixturePrior(prior_folder=args().smpl_model_path,num_gaussians=8,dtype=torch.float32).cuda()
        if args().HMloss_type=='focal':
            args().heatmap_weight /=1000
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.joint_lossweights = torch.from_numpy(constants.SMPL54_weights).float()
        self.align_inds_MPJPE = np.array([constants.SMPL_ALL_54['L_Hip'], constants.SMPL_ALL_54['R_Hip']])
        self.shape_pca_weight = torch.Tensor([1, 0.32, 0.16, 0.16, 0.08, 0.08, 0.08, 0.04, 0.02, 0.01]).unsqueeze(0).float()
        self.test_train=args().test_train
    def forward(self, outputs, **kwargs):
        meta_data = outputs['meta_data']
        detect_loss_dict = self._calc_detection_loss(outputs, meta_data)
        detection_flag = outputs['detection_flag'].sum() if args().model_return_loss else outputs['detection_flag']
        if detection_flag or args().model_return_loss:
            kp_loss_dict, kp_error = self._calc_keypoints_loss(outputs, meta_data) #, kp_acc_dict
            params_loss_dict = self._calc_param_loss(outputs, meta_data)
            #bbox_loss_dict=self._calc_bbox_loss(outputs, meta_data)
            hvip_loss_dict=self._calc_hvip_loss(outputs, meta_data)
            #ankle2d_loss_dict=self._calc_ankle_loss(outputs, meta_data)
            # ankle2d_loss_dict=self._calc_ankle_in_image_loss(outputs, meta_data)
            # delta2d_loss_dict=self._calc_delta2d_loss(outputs, meta_data)
            # trans3d_loss_dict=self._calc_trans_loss(outputs, meta_data)
            root_cam_loss_dict=self._calc_root_cam_loss(outputs, meta_data)
            out_of_bound_loss_dict=self._calc_out_of_bound_loss(outputs, meta_data)
            tcs_offset_loss_dict=self._calc_tcs_offset_loss(outputs, meta_data)
            ground_normal_loss_dict=self._calc_ground_normal_loss(outputs, meta_data)

            loss_dict = dict(detect_loss_dict, **kp_loss_dict, **params_loss_dict, **hvip_loss_dict,  **out_of_bound_loss_dict, **root_cam_loss_dict, **tcs_offset_loss_dict, **ground_normal_loss_dict) # **bbox_loss_dict
            # if self.test_train:
            #     bend_leg_loss=self._calc_bend_leg_loss(outputs, meta_data)
            #     loss_dict.update(bend_leg_loss)
        else:
            loss_dict = detect_loss_dict
            kp_error = None

        loss_names = list(loss_dict.keys())
        for name in loss_names:
            if isinstance(loss_dict[name],tuple):
                loss_dict[name] = loss_dict[name][0]
            elif isinstance(loss_dict[name],int):
                loss_dict[name] = torch.zeros(1,device=outputs['center_map'].device)
            loss_dict[name] = loss_dict[name].mean() * eval('args().{}_weight'.format(name))

        return {'loss_dict':loss_dict, 'kp_error':kp_error}

    def _calc_detection_loss(self, outputs, meta_data):
        device = outputs['center_map'].device
        detect_loss_dict = {'CenterMap': 0}
        all_person_mask = meta_data['all_person_detected_mask'].to(device)
        if all_person_mask.sum()>0:
            detect_loss_dict['CenterMap'] = focal_loss(outputs['center_map'][all_person_mask], meta_data['centermap'][all_person_mask].to(device)) #((centermaps-centermaps_gt)**2).sum(-1).sum(-1).mean(-1) #
        return detect_loss_dict

    def _calc_bbox_loss(self, outputs, meta_data):
        bbox_loss_dict={'full_bbox':0}

        if 'full_bbox' in outputs:
            real_2d = meta_data['full_bbox2d'].to(outputs['full_bbox'].device)
            bbox_loss_dict['full_bbox']=batch_bbox_2d_l2_loss(real_2d.float(), outputs['full_bbox'].float())
        return bbox_loss_dict
    
    def _calc_hvip_loss(self, outputs, meta_data):
        hvip_loss={'hvip2d':0}
        hvip_mask = meta_data['valid_masks'][:, 6]
        if hvip_mask.sum()>1  and 'old_hvip2d' in outputs:
            real_hvip=meta_data['hvip2ds'][hvip_mask].contiguous().to(outputs['old_hvip2d'].device)
            pred_hvip=outputs['old_hvip2d'][hvip_mask].contiguous()
            hvip_loss['hvip2d']=batch_bbox_2d_l2_loss(real_hvip.float(), pred_hvip.float())
        return hvip_loss

    def _calc_out_of_bound_loss(self, outputs, meta_data):
        side_ground_loss={'out_of_bound':0}
        if 'dist_ground' in outputs:
            side_ground_loss['out_of_bound']=compute_side_ground_loss(outputs['dist_ground'])
        return side_ground_loss

    def _calc_delta2d_loss(self, outputs, meta_data):
        delta2d_loss={'delta2d':0}
        delta2d_mask = meta_data['valid_masks'][:, 6]
        if delta2d_mask.sum()>1  and 'predict_delta2d' in outputs:
            real_delta2d=meta_data['delta2d'][delta2d_mask].contiguous().to(outputs['predict_delta2d'].device)
            preds_delta2d=outputs['predict_delta2d'][delta2d_mask].contiguous()
            delta2d_loss['delta2d']=batch_bbox_2d_l2_loss(real_delta2d.float(), preds_delta2d.float())
            # print('real_delta2d',real_delta2d.shape, 'preds_delta2d', preds_delta2d.shape)
        return delta2d_loss
    
    def _calc_ankle_loss(self, outputs, meta_data):
        ankle_loss={'ankle2d':0}
        ankle_index=[7,8]
        ankle_mask = meta_data['valid_masks'][:, 8]
        if ankle_mask.sum()>1  and 'predict_ankle2d' in outputs:
            real_2d = meta_data['full_kp2d'][ankle_mask].contiguous().to(outputs['predict_ankle2d'].device)# m, 54, 2
            real_ankle2d=real_2d[:, ankle_index].reshape(-1, 4) # m, 4

            preds_ankle2d=outputs['predict_ankle2d'][ankle_mask].contiguous().reshape(-1, 4) # m, 4
            ankle_loss['ankle2d']=batch_bbox_2d_l2_loss(real_ankle2d.float(), preds_ankle2d.float())

        return ankle_loss
    
    def convert_ankle_in_image(self, ankles):
        '''
        intput: ankles (m, 2, 2)
        output: ankles_in_image (m, 2, 2)
        '''
        ankles_in_image=ankles.clone()
        for i in range(ankles.shape[0]):
            if ankles[i][0][0]>ankles[i][1][0]:
                ankles_in_image[i][0]=ankles[i][1]
                ankles_in_image[i][1]=ankles[i][0]
        return ankles_in_image
    
    def _calc_ankle_in_image_loss(self, outputs, meta_data): # predict left, right ankle in image
        ankle_loss={'ankle2d':0}
        ankle_mask = meta_data['valid_masks'][:, 8]
        if ankle_mask.sum()>1  and 'predict_ankle2d' in outputs:
            real_ankle2d = meta_data['ankle2ds'][ankle_mask].contiguous().to(outputs['predict_ankle2d'].device)# m, 2, 2
            real_ankle2d_in_image=self.convert_ankle_in_image(real_ankle2d)
            # print('ankle:',real_ankle2d.detach().cpu().numpy())
            #print('ankle_in_image', real_ankle2d_in_image.detach().cpu().numpy())
            preds_ankle2d=outputs['predict_ankle2d'][ankle_mask].contiguous().reshape(-1, 4) # m, 4
            ankle_loss['ankle2d']=batch_bbox_2d_l2_loss(real_ankle2d_in_image.reshape(-1, 4).float(), preds_ankle2d.float())

        return ankle_loss


    def _calc_trans_loss(self, outputs, meta_data):
        trans_loss={'trans3d':0}
        trans_mask = meta_data['valid_masks'][:, 7]
        if trans_mask.sum()>1  and 'pred_trans' in outputs:
            real_trans=meta_data['trans3d'][trans_mask].contiguous().to(outputs['old_hvip2d'].device)
            pred_trans=outputs['pred_trans'][trans_mask].contiguous()
            trans_loss['trans3d']=batch_bbox_2d_l2_loss(real_trans.float(), pred_trans.float())
        return trans_loss
    
    def _calc_root_cam_loss(self, outputs, meta_data):
        root_cam_loss={'root_cam':0}
        root_cam_mask = meta_data['valid_masks'][:, 9]
        if root_cam_mask.sum()>1  and 'pred_root_cam' in outputs:
            real_root_cam=meta_data['root_cam'][root_cam_mask].contiguous().to(outputs['pred_root_cam'].device)
            pred_root_cam=outputs['pred_root_cam'][root_cam_mask].contiguous()
            root_cam_loss['root_cam']=batch_bbox_2d_l2_loss(real_root_cam.float(), pred_root_cam.float())
        return root_cam_loss

    def _calc_tcs_offset_loss(self, outputs, meta_data):
        tc_offset_loss={'tc_offset':0}
        if 'torso_center' in outputs:
            real_tc = meta_data['real_tcs'].contiguous().to(outputs['torso_center'].device)
            torso_center=outputs['torso_center'].contiguous()
            tc_offset_loss['tc_offset']=batch_bbox_2d_l2_loss(real_tc.float(), torso_center.float())
        return tc_offset_loss
 

    def _calc_keypoints_loss(self, outputs, meta_data):
        kp_loss_dict, error = {'P_KP2D':0, 'MPJPE':0, 'PAMPJPE':0}, {'3d':{'error':[], 'idx':[]},'2d':{'error':[], 'idx':[]}}
        if 'pj2d' in outputs:
            real_2d = meta_data['full_kp2d'].to(outputs['pj2d'].device)
            #print('real_2d', real_2d.cpu().detach().numpy())
            #print('output_2d', outputs['pj2d'].cpu().detach().numpy())
            if args().model_version == 3:
                kp_loss_dict['joint_sampler'] = self.joint_sampler_loss(real_2d, outputs['joint_sampler_pred']) # ?
            kp_loss_dict['P_KP2D'] = batch_kp_2d_l2_loss(real_2d.float(), outputs['pj2d'][:, :54].float(), weights=self.joint_lossweights)
        
            kp3d_mask = meta_data['valid_masks'][:,1]#.to(outputs['j3d'].device)
            if (~kp3d_mask).sum()>1:
                error['2d']['error'].append(kp_loss_dict['P_KP2D'][~kp3d_mask].detach()*1000) # have 2d without 3d
                error['2d']['idx'].append(torch.where(~kp3d_mask)[0])

        if kp3d_mask.sum()>1 and 'j3d' in outputs:
            kp3d_gt = meta_data['kp_3d'][kp3d_mask].contiguous().to(outputs['j3d'].device)
            preds_kp3d = outputs['j3d'][kp3d_mask, :kp3d_gt.shape[1]].contiguous()
            
            if args().MPJPE_weight>0:
                mpjpe_each = calc_mpjpe(kp3d_gt, preds_kp3d, align_inds=self.align_inds_MPJPE)
                kp_loss_dict['MPJPE'] = mpjpe_each
                error['3d']['error'].append(mpjpe_each.detach()*1000)
                error['3d']['idx'].append(torch.where(kp3d_mask)[0])
            if not args().model_return_loss and args().PAMPJPE_weight>0 and len(preds_kp3d)>0:
                try:
                    pampjpe_each = calc_pampjpe(kp3d_gt.contiguous(), preds_kp3d.contiguous())
                    kp_loss_dict['PAMPJPE'] = pampjpe_each
                except Exception as exp_error:
                    print('PA_MPJPE calculation failed!', exp_error)

        return kp_loss_dict, error

    # def _calc_keypoints_loss_wh(self, outputs, meta_data):
    #     kp_loss_dict, error = {'P_KP2D': 0, 'MPJPE': 0, 'PAMPJPE': 0}, {'3d': {'error': [], 'idx': []},
    #                                                                     '2d': {'error': [], 'idx': []}}
    #     if 'pj2d' in outputs:
    #         real_2d = meta_data['full_kp2d'].to(outputs['pj2d'].device)
    #         if args().model_version == 3:
    #             kp_loss_dict['joint_sampler'] = self.joint_sampler_loss(real_2d, outputs['joint_sampler_pred'])  # ?
    #         kp_loss_dict['P_KP2D'] = batch_kp_2d_l2_loss(real_2d.float(), outputs['pj2d'].float(),
    #                                                      weights=self.joint_lossweights)
    #
    #         kp3d_mask = meta_data['valid_masks'][:, 1]  # .to(outputs['j3d'].device)
    #         if (~kp3d_mask).sum() > 1:
    #             error['2d']['error'].append(kp_loss_dict['P_KP2D'][~kp3d_mask].detach() * 1000)  # have 2d without 3d
    #             error['2d']['idx'].append(torch.where(~kp3d_mask)[0])
    #
    #     if kp3d_mask.sum() > 1 and 'j3d' in outputs:
    #         kp3d_gt = meta_data['kp_3d'][kp3d_mask].contiguous().to(outputs['j3d'].device)
    #         preds_kp3d = outputs['j3d'][kp3d_mask, :kp3d_gt.shape[1]].contiguous()
    #
    #         if args().MPJPE_weight > 0:
    #             mpjpe_each = calc_mpjpe(kp3d_gt, preds_kp3d, align_inds=self.align_inds_MPJPE)
    #             kp_loss_dict['MPJPE'] = mpjpe_each
    #             error['3d']['error'].append(mpjpe_each.detach() * 1000)
    #             error['3d']['idx'].append(torch.where(kp3d_mask)[0])
    #         if not args().model_return_loss and args().PAMPJPE_weight > 0 and len(preds_kp3d) > 0:
    #             try:
    #                 pampjpe_each = calc_pampjpe(kp3d_gt.contiguous(), preds_kp3d.contiguous())
    #                 kp_loss_dict['PAMPJPE'] = pampjpe_each
    #             except Exception as exp_error:
    #                 print('PA_MPJPE calculation failed!', exp_error)
    #
    #     return kp_loss_dict, error

    def _calc_param_loss(self, outputs, meta_data):
        params_loss_dict = {'Pose': 0, 'Shape':0, 'Prior':0}

        if 'params' in outputs:
            _check_params_(meta_data['params'])
            device = outputs['params']['body_pose'].device
            grot_masks, smpl_pose_masks, smpl_shape_masks = meta_data['valid_masks'][:,3].to(device), meta_data['valid_masks'][:,4].to(device), meta_data['valid_masks'][:,5].to(device)

            if grot_masks.sum()>0:
                params_loss_dict['Pose'] += batch_l2_loss_param(meta_data['params'][grot_masks,:3].to(device).contiguous(), outputs['params']['global_orient'][grot_masks].contiguous()).mean()

            if smpl_pose_masks.sum()>0:
                params_loss_dict['Pose'] += batch_l2_loss_param(meta_data['params'][smpl_pose_masks,3:22*3].to(device).contiguous(), outputs['params']['body_pose'][smpl_pose_masks,:21*3].contiguous()).mean()

            if smpl_shape_masks.sum()>0: # 1 ?
                # beta annots in datasets are for each gender (male/female), not for our neutral. 
                smpl_shape_diff = meta_data['params'][smpl_shape_masks,-10:].to(device).contiguous() - outputs['params']['betas'][smpl_shape_masks,:10].contiguous()
                params_loss_dict['Shape'] += torch.norm(smpl_shape_diff, p=2, dim=-1).mean() / 20.

            # Don't ask to force the first 2-dim body scale/fat value to be 0. Let it learn from body age/type.
            if (~smpl_shape_masks).sum()>0:
                params_loss_dict['Shape'] += (outputs['params']['betas'][~smpl_shape_masks,1:10]**2).mean() / 10.

            gmm_prior_loss = self.gmm_prior(outputs['params']['body_pose'], outputs['params']['betas']).mean()/100.
            angle_prior_loss = angle_prior(outputs['params']['body_pose']).mean()/5.
            params_loss_dict['Prior'] = gmm_prior_loss + angle_prior_loss

        return params_loss_dict

    def joint_sampler_loss(self, real_2d, joint_sampler):
        batch_size = joint_sampler.shape[0]
        joint_sampler = joint_sampler.view(batch_size, -1, 2)
        joint_gt = real_2d[:,constants.joint_sampler_mapper]
        loss = batch_kp_2d_l2_loss(joint_gt, joint_sampler)
        return loss
    
    def _calc_ground_normal_loss(self, outputs, meta_data):
        ground_normal_loss={'ground_normal': 0}
        ankle_index=[7, 8]
        shouder_index=[16, 17]
        if 'j3d' in outputs and 'ground' in meta_data:
            j3ds=outputs['j3d']
            ground=meta_data['ground']
            ground_N=ground[:, :3]
            ankles=j3ds[:, ankle_index, :]
            shouders=j3ds[:, shouder_index, :]
            ankles_mean=ankles.mean(1)
            shouders_mean=shouders.mean(1)
            person_direct=shouders_mean - ankles_mean
            similarity=torch.cosine_similarity(person_direct, ground_N, dim=1)
            # cos_v=self.compute_cos(person_direct, ground_N)
            loss=(1-similarity).mean()
            ground_normal_loss={'ground_normal': loss}
            
        return ground_normal_loss
    
    def _calc_bend_leg_loss(self, outputs, meta_data):
        bend_leg_loss={'bend_leg': 0}
        if 'j3d' in outputs :
            j3ds=outputs['j3d']

            left_ankle=j3ds[:, constants.SMPL_ALL_54['L_Ankle']]
            left_knee=j3ds[:, constants.SMPL_ALL_54['L_Knee']]
            left_hip=j3ds[:, constants.SMPL_ALL_54['L_Hip']]

            right_ankle=j3ds[:, constants.SMPL_ALL_54['R_Ankle']]
            right_knee=j3ds[:, constants.SMPL_ALL_54['R_Knee']]
            right_hip=j3ds[:, constants.SMPL_ALL_54['R_Hip']]

            left_loss=_blend_cost(left_ankle, left_knee, left_hip)
            right_loss=_blend_cost(right_ankle, right_knee, right_hip)

            bend_leg_loss={'bend_leg': left_loss + right_loss}
            
        return bend_leg_loss

def _check_params_(params):
    assert params.shape[0]>0, logging.error('meta_data[params] dim 0 is empty, params: {}'.format(params))
    assert params.shape[1]>0, logging.error('meta_data[params] dim 1 is empty, params shape: {}, params: {}'.format(params.shape, params))

def _blend_cost(ankle, knee, hip, tre=1):
    v1=knee - hip
    v2=ankle - knee
    similarity=torch.cosine_similarity(v1, v2, dim=1)
    similarity[similarity>tre]=1.
    similarity[similarity<0]=1.
    loss=1 - similarity
    return loss.mean()