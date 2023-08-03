import os
import numpy as np
from lib.utils import read_pkl, write_json
from lib.merge_utils import merge_return_cid, extract_results_after_merged
from lib.smpl import SMPL
from natsort import natsorted
import torch

from global_setting import *
smpl_J_reg_extra_path=os.path.join(smpl_model_root, 'J_regressor_extra.npy')
smpl_n=SMPL(smpl_model_root, J_reg_extra9_path=smpl_J_reg_extra_path,  gender='neutral', align_by_image_pelvis=False) 
torso_index=[16, 17, 45, 46]

def foot2d_to_foot3d(foot2d_in_scene, camM, ground):
    '''
    foot2d_in_scene   (m,2)
    camM   (3,3)
    ground   (4,)
    '''
    foot2d_in_scene = np.concatenate([foot2d_in_scene, np.ones((foot2d_in_scene.shape[0], 1))], axis=-1) # m, 3
    fx_inverse, fy_inverse = 1/camM[0,0], 1/camM[1, 1]
    new_depth = -ground[3] / (
            ground[0] * fx_inverse * (foot2d_in_scene[:, 0] - camM[0, 2]) \
            + ground[1] * fy_inverse * (foot2d_in_scene[:, 1] - camM[1, 2]) \
            + ground[2])
    K_inv = np.linalg.inv(camM)
    temp = foot2d_in_scene * np.expand_dims(new_depth, -1)
    foot3d = (K_inv @ temp.T).T # m,3
    return foot3d

def compute_quantative_for_submit(inference_file, sceneImage_root,  ground_root, scene_types=[], mid_root='crowd_midResults/', save_json_folder='', merge_mode='tc'):

    
    results_for_eval={}
    for scene_type in scene_types:
        frame_list=os.listdir(os.path.join(sceneImage_root, scene_type))
        inference_path=os.path.join(inference_file, scene_type)
        all_result_path = os.path.join(inference_path, 'all_result.pkl')
        all_result_orign=read_pkl(all_result_path)

        for scene_name in natsorted(frame_list):
            # for our model pred
            scene_pre=scene_name.replace('.jpg', '')
            all_result=all_result_orign.copy()
            ground_path=os.path.join(ground_root, scene_type)
            ground=np.load(os.path.join(ground_path, 'ground.npy'))
            camK=np.load(os.path.join(ground_path, 'cam_para.npy'))
            position_path = os.path.join(mid_root, scene_type, scene_pre, 'position.npy')
            row_col_path = os.path.join(mid_root, scene_type, scene_pre, 'row_col.npy')
            

            final_center_points, final_center_points_xy, total_cps_num, cps2id_dict, x2row_dict, y2col_dict = merge_return_cid(
                position_path, row_col_path,
                scene_pre, all_result, mode=merge_mode)

            merge_result=extract_results_after_merged(final_center_points,final_center_points_xy,cps2id_dict, all_result,x2row_dict,y2col_dict, scene_pre)
            results_for_eval[scene_name]={
            'Intrinsic_matrix': camK.tolist(), 
            'person_list':[]
            }

            for patch_name in merge_result:
                cur_data=merge_result[patch_name]
                for person_dict in cur_data:
                    final_trans=person_dict['final_trans']
                    poses=person_dict['pose']
                    betas=person_dict['betas']
                    foot2d_in_scene=person_dict['hvip2d_in_scene']
                   
                    poses_torch=torch.from_numpy(np.array(poses).reshape(1, 72))
                    betas_torch=torch.from_numpy(np.array(betas).reshape(1, 10))
                    smpl_out=smpl_n(poses=poses_torch, betas=betas_torch)
                    verts, j3ds=smpl_out['verts'].numpy()[0], smpl_out['j3d'].numpy()[0]
                    pelvis_root=j3ds[49]
                    final_trans-=pelvis_root

                    # process trans for competition eval
                    j3ds_cam=j3ds + np.expand_dims(final_trans, 0)
                    torso_root_cam=np.expand_dims(j3ds_cam[torso_index].mean(0), 0)
                    torso_root_smpl=np.expand_dims(j3ds[torso_index].mean(0), 0)
                    j3ds_cam_new = (j3ds - torso_root_smpl) * 1 + torso_root_cam
                    j3ds_smpl_new = j3ds * 1
                    diff=(j3ds_cam_new - j3ds_smpl_new).mean(0)
                    final_trans=diff

                    poses, betas, final_trans=poses.astype(np.float64), betas.astype(np.float64), final_trans.astype(np.float64)
                    
                    # add hvip and HVIP
                    if foot2d_in_scene is None:
                        print('hvip2d_in_scene', foot2d_in_scene)
                        exit()
  
                    hvip=list(foot2d_in_scene.astype(np.float64))
                    LP=foot2d_to_foot3d(foot2d_in_scene.reshape(1, 2), camK, ground)
                    HVIP=list(LP[0].astype(np.float64))

                    eval_person_dict={'pose':list(poses), 'shape':list(betas), 'trans_cam': list(final_trans), 'gender':'neutral', 'scale_smpl': 1., 'hvip2d':hvip, 'hvip3d':HVIP}
                    results_for_eval[scene_name]['person_list'].append(eval_person_dict)
                
    if save_json_folder != '':
        os.makedirs(save_json_folder, exist_ok=True)
        write_json(os.path.join(save_json_folder, 'predict.json'), results_for_eval)
        print('save as', os.path.join(save_json_folder, 'predict.json'))

    return