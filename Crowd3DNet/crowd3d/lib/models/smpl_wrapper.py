import torch
import torch.nn as nn
import numpy as np 

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
from config import args
import constants
from models.smpl import SMPL
from utils.projection import vertices_kp3d_projection, vertices_kp3d_projection_ground
from utils.rot_6D import rot6D_to_angular

class SMPLWrapper(nn.Module):
    def __init__(self):
        super(SMPLWrapper,self).__init__()
        self.smpl_model = SMPL(args().smpl_model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, \
            batch_size=args().batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False).cuda()
        self.part_name = ['hvip2d', 'tc_offset',  'delta3d', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [args().hvip_dim, 2,  3, args().rot_dim,  (args().smpl_joint_num-1)*args().rot_dim,       10]
        self.unused_part_name = ['left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
        self.unused_part_idx = [        15,                  15,           3,          3,            3,          10]
        
        self.kps_num = 25 # + 21*2
        # self.params_num = np.array(self.part_idx).sum()
        self.params_num =  args().rot_dim + (args().smpl_joint_num-1)*args().rot_dim + 10
        self.global_orient_nocam = torch.from_numpy(constants.global_orient_nocam).unsqueeze(0)
        self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()
        self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()

    def forward(self, outputs, meta_data):
        idx_list, params_dict = [0], {}
        for i,  (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = outputs['params_pred'][:, idx_list[i]: idx_list[i+1]].contiguous()
        # for k,v in params_dict.items():
        #     print(k, v.shape)
        # print('params_pred', outputs['params_pred'].shape)
        if args().Rot_type=='6D':
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient']) 
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N,6).to(params_dict['body_pose'].device)],1) 
        params_dict['poses'] = torch.cat([params_dict['global_orient'], params_dict['body_pose']], 1)
 
        smpl_outs = self.smpl_model(**params_dict, return_verts=True, return_full_pose=True)

        outputs.update({'params': params_dict, **smpl_outs})
        outputs.update(vertices_kp3d_projection_ground(outputs, meta_data=meta_data))
        return outputs