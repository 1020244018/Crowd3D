import os
from natsort import natsorted
import numpy as np
from single_image_configs import *
from lib.part_utils import part_image
from lib.pose_inference import pose_inference
from lib.generate_test_train import generate_test_train_scale
from lib.merge_joints import merge_joints
from lib.filter_joints import filter_kps
from lib.visualization import vis_joint_2d
from lib.generate_ground import generate_ground, trans_ground
from lib.simplify_utils import is_people, scale_patches
from lib.utils import read_json
from global_setting import *
import argparse

parser=argparse.ArgumentParser(description='process single image')
parser.add_argument('--scene_image_path', type=str, default='')
# parser.add_argument('--close_visual_merge_joints', action='store_true', default=True)
parser.add_argument('--save_mid_folder', type=str, default='')
args=parser.parse_args()

flag_dict={
    'part_flag': True,
    'pose_inference': True,
    'generate_ground': True
}

if args.scene_image_path == '':
    print('Please input the scene_image_path')
    exit()

scene_image_path=args.scene_image_paths
save_mid_folder=args.save_mid_folder
visual_merge_joints=True # Visualize the merge joints
visual_part=True # Visualization of block results
use_scale_blocks=True # True is ok.

single_part_params=read_json('params/single_image_part_params.json')
name_pre=os.path.basename(scene_image_path).replace('.jpg', '').replace('.png', '')
if save_mid_folder== '':
    save_mid_folder=os.path.join(project_root, 'single_image', 'single_image_mid', name_pre)
os.makedirs(save_mid_folder, exist_ok=True)

print('Save as %s' %save_mid_folder)

# 1. part
if flag_dict['part_flag']:
    print('1. part image')
    scene_part_params=single_part_params[os.path.basename(scene_image_path)]
    part_image(scene_image_path, save_mid_folder, \
                scene_part_params[0], scene_part_params[1], scene_part_params[2], scene_part_params[3], cor=2, save_orign_blocks=False, save_scale_blocks=True, visual=visual_part, cover=True)


# 2. pose inference # need alphapose environment
save_merge_joints_name=os.path.join(save_mid_folder, 'joints_2d_alphapose_merge.npy')
save_merge_filter_joints_name=os.path.join(save_mid_folder, 'joints_2d_alphapose_merge_filter.npy')
if flag_dict['pose_inference']:
    print('2. pose inference')
    if use_scale_blocks:
        pose_inference_input_path=os.path.join(save_mid_folder, 'part_images_scale')
    else:
        pose_inference_input_path=os.path.join(save_mid_folder, 'part_images')

    if flag_dict['pose_inference']:
        pose_inference(pose_inference_input_path, save_mid_folder, alphapose_root, save_image=False, use_scale_blocks=use_scale_blocks)

        # 2d pose merge and filter
        # merge
        joints_2d=merge_joints(save_mid_folder, save_mid_folder, name_pre, use_scale_blocks=use_scale_blocks)
        joints_2d_merge=np.array(joints_2d)
        np.save(save_merge_joints_name, joints_2d_merge)

        # filter joints
        new_joints_2d=filter_kps(joints_2d)
        # print('after filter %d -> %d' %(len(joints_2d), len(new_joints_2d)))
        new_joints_2d=np.array(new_joints_2d)
        np.save(save_merge_filter_joints_name, new_joints_2d)
        if visual_merge_joints:
            # save_check_joints_name=os.path.join(pose_inference_output_path, 'joints_2d_merge.jpg')
            # vis_joint_2d(frame_path, joints_2d_merge, save_check_joints_name)
            if use_scale_blocks:
                save_check_joints_name=os.path.join(save_mid_folder, 'joints_2d_merge_filter_scale.jpg')
            else:
                save_check_joints_name=os.path.join(save_mid_folder, 'joints_2d_merge_filter.jpg')
            vis_joint_2d(scene_image_path, new_joints_2d, save_check_joints_name)

        # have people by pose
        is_people(pose_inference_input_path, joints_2d_merge, os.path.join(save_mid_folder, 'have_people_image_name.json'))
        # generate annots for optim
        generate_test_train_scale(save_mid_folder, name_pre, test_and_visual=True)



# 3. generate_ground
save_ground_path=os.path.join(save_mid_folder, 'ground_mid', 'mid')
save_ground_trans_path=os.path.join(save_mid_folder, 'ground_mid', 'ground')
if flag_dict['generate_ground']:
    print('3. generate ground.')
    generate_ground(save_merge_filter_joints_name, scene_image_path, save_ground_path, cwd=estimate_ground_root)
    
    ground=np.load(os.path.join(save_ground_path, 'ground.npy'))
    cam=np.load(os.path.join(save_ground_path, 'cam_para.npy'))
    scene_shape=np.load(os.path.join(save_ground_path, 'scene_shape.npy'))
    # trans ground 0.1m
    trans_for_ground=0.1
    ground_trans=trans_ground(ground, distant_to_ground=trans_for_ground)
    os.makedirs(save_ground_trans_path, exist_ok=True)
    np.save(os.path.join(save_ground_trans_path, 'ground.npy'), ground_trans)
    np.save(os.path.join(save_ground_trans_path, 'cam_para.npy'), cam)
    np.save(os.path.join(save_ground_trans_path, 'scene_shape.npy'), scene_shape) # w, h
