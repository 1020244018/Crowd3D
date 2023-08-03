import os, json, cv2
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import time
from test_configs import *
from test_constants import *
from lib.part_utils import part_image
from lib.pose_inference import pose_inference
from lib.generate_test_train import generate_test_train_scale
from lib.merge_joints import merge_joints
from lib.filter_joints import filter_kps
from lib.visualization import vis_joint_2d
from lib.generate_ground import generate_ground, trans_ground
from lib.simplify_utils import is_people
from lib.utils import read_json
from global_setting import *
import argparse



part_params=read_json(os.path.join(project_root, 'params/part_params_crowd.json'))
image_folder=os.path.join(project_root, 'data/largecrowd/images/test/')
save_root=os.path.join(project_root, 'test', 'test_mid') 
ground_save_root=os.path.join(save_root, 'ground_mid')
os.makedirs(save_root, exist_ok=True)

process_scenes=eval_scenes

flag_dict={
    'part_flag': True,
    'pose_inference': True,
    'generate_ground': True
}
use_scale_blocks=True # stop using.
visual_merge_joints=True # for alphapsoe merge result on selectframe.

print('Save as %s' % save_root)

# 1. part all images
print('step1: crop image into patchs')
for scene in process_scenes:
    frame_list=natsorted(os.listdir(os.path.join(image_folder, scene)))
    scene_part_params=part_params[scene]
    for frame_name in tqdm(frame_list):
        name_pre=frame_name.replace('.jpg', '')
        frame_path=os.path.join(image_folder, scene, frame_name)
        frame_save_folder=os.path.join(save_root, scene, frame_name.replace('.jpg', ''))
        # 1. part
        time1=time.time()
        if flag_dict['part_flag']:
            part_image(os.path.join(image_folder, scene, frame_name), frame_save_folder, \
                scene_part_params[0], scene_part_params[1], scene_part_params[2], scene_part_params[3], cor=2, save_orign_blocks=False, save_scale_blocks=use_scale_blocks, scale_size=512, visual=False)



# 2. pose inference
print('step2: 2D pose inference and process for selectframe')
for scene in process_scenes:
    for frame_name in select_frame_dict[scene]: 
        frame_save_folder=os.path.join(save_root, scene, frame_name.replace('.jpg', ''))
        if use_scale_blocks:
            pose_inference_input_path=os.path.join(frame_save_folder, 'part_images_scale')
        else:
            pose_inference_input_path=os.path.join(frame_save_folder, 'part_images')
        pose_inference_output_path=frame_save_folder
        save_merge_joints_name=os.path.join(pose_inference_output_path, 'joints_2d_alphapose_merge.npy')
        save_merge_filter_joints_name=os.path.join(pose_inference_output_path, 'joints_2d_alphapose_merge_filter.npy')
        if flag_dict['pose_inference']:
            pose_inference(pose_inference_input_path, pose_inference_output_path, alphapose_root, save_image=False, use_scale_blocks=use_scale_blocks)

            # 2d pose merge and filter
            # merge
            joints_2d=merge_joints(pose_inference_output_path, frame_save_folder, frame_name.replace('.jpg', ''), use_scale_blocks=use_scale_blocks)
            joints_2d_merge=np.array(joints_2d)
            np.save(save_merge_joints_name, joints_2d_merge)

            # filter joints
            new_joints_2d=filter_kps(joints_2d)
            # print('after filter %d -> %d' %(len(joints_2d), len(new_joints_2d)))
            new_joints_2d=np.array(new_joints_2d)
            np.save(save_merge_filter_joints_name, new_joints_2d)
            if visual_merge_joints:
                frame_path=os.path.join(image_folder, scene, frame_name)
                # save_check_joints_name=os.path.join(pose_inference_output_path, 'joints_2d_merge.jpg')
                # vis_joint_2d(frame_path, joints_2d_merge, save_check_joints_name)
                save_check_joints_name=os.path.join(pose_inference_output_path, 'joints_2d_merge_filter.jpg')
                vis_joint_2d(frame_path, new_joints_2d, save_check_joints_name)
            
            # have people by pose
            is_people(pose_inference_input_path, joints_2d_merge, os.path.join(frame_save_folder, 'have_people_image_name.json'))
            # generate annots for optim
            generate_test_train_scale(frame_save_folder, frame_name, test_and_visual=False)


        

# # 3. generate_ground
print('step3: search ground')
ground_pre='ground'
fps_sample_num=-1 # sample xx person to predict ground plane
if flag_dict['generate_ground']:
    for scene in process_scenes:
        for frame_name in select_frame_dict[scene]:
            frame_save_folder=os.path.join(save_root, scene, frame_name.replace('.jpg', ''))
            frame_path=os.path.join(image_folder, scene, frame_name)
            
            save_merge_filter_joints_name=os.path.join(frame_save_folder, 'joints_2d_alphapose_merge_filter.npy')
            save_ground_path=os.path.join(ground_save_root, 'mid',  scene, frame_name.replace('.jpg', ''))
            
            if fps_sample_num>0:
                ground_pre='ground_sample_'+str(fps_sample_num)
            generate_ground(save_merge_filter_joints_name, frame_path, save_ground_path, fps_sample_num=fps_sample_num, cwd=estimate_ground_root)
            ground=np.load(os.path.join(save_ground_path, 'ground.npy'))
            cam=np.load(os.path.join(save_ground_path, 'cam_para.npy'))
            scene_shape=np.load(os.path.join(save_ground_path, 'scene_shape.npy'))

            # trans ground 0.1m
            trans_for_ground=0.1
            ground_trans=trans_ground(ground, distant_to_ground=trans_for_ground)
            
            save_ground_trans_path=os.path.join(ground_save_root, ground_pre, scene)
            os.makedirs(save_ground_trans_path, exist_ok=True)
            np.save(os.path.join(save_ground_trans_path, 'ground.npy'), ground_trans)
            np.save(os.path.join(save_ground_trans_path, 'cam_para.npy'), cam)
            np.save(os.path.join(save_ground_trans_path, 'scene_shape.npy'), scene_shape) # w, h