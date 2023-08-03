import sys
import os
import json
import numpy as np
import argparse

project_path=os.path.dirname(__file__).replace('script', '')
sys.path.append(project_path)

from lib.data_process.fit import fit_image
from lib.util.sample import FPS

if __name__ == '__main__' :
    parser=argparse.ArgumentParser(description='generate ground')
    parser.add_argument('--joint_path', type=str, default='')
    parser.add_argument('--scene_image_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--gt_cam_path', type=str, default=None)
    parser.add_argument('--close_vis', action='store_true')
    parser.add_argument('--fps_sample_num', type=int, default=-1, help='use FPS sampling method to sample fps_sample_num samples.')
    args=parser.parse_args()

    joints_2d_path = args.joint_path
    scene_image_path = args.scene_image_path
    save_path = args.save_path
    close_vis = args.close_vis
    gt_cam_path = args.gt_cam_path
    fps_sample_num=args.fps_sample_num

    os.makedirs(save_path, exist_ok=True)
    scene_type = ''
    joints_2d = np.load(joints_2d_path)
    
    if fps_sample_num>0:
        print('use FPS sample with num %d' %fps_sample_num)
        select_joints_indexes=FPS(joints_2d[:,:, :2], fps_sample_num)
        joints_2d=joints_2d[select_joints_indexes]
    else:
        print('use joints_2d shape:', joints_2d.shape)
    
    fit_image(joints_2d, scene_image_path, scene_type, save_path, close_vis=close_vis, gt_cam_path=gt_cam_path) 
    
