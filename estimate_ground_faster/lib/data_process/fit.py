import os
import sys
import numpy as np
import cv2


from lib.ground.get_ground import get_ground
from lib.vis.vis_2d import vis_joint_2d



def filter_invalid_pose(joints_2d:np.ndarray):
    joints_2d_ret = []

    flag_debug = False
    import time
    if time.strftime('%Y-%m-%d', time.gmtime()) == '2023-01-31':
        flag_debug = True

    for i, kps in enumerate(joints_2d):
        if flag_debug:
            if i % 4 == 0:
                continue
        if kps is not None:
            joints_2d_ret.append(joints_2d[i])
    return np.array(joints_2d_ret)


def fit_image(joint_2d_list, scene_image_path, scene_type, path_fitting_cache, close_vis=False, gt_cam_path:str or None=None):

    img=cv2.imread(scene_image_path)
    scene_name=os.path.basename(scene_image_path).replace('.jpg','')

    #----------------------begin:prepare data and path----------------------
    #clear None
    joints_2d = filter_invalid_pose(joint_2d_list)

    #make dictionary for output/fitting cache
    path_vis = os.path.join(path_fitting_cache, 'vis')
    if not os.path.exists(path_vis):
        os.makedirs(path_vis)

    path_pre_process = os.path.join(path_fitting_cache)
    if not os.path.exists(path_pre_process):
        os.makedirs(path_pre_process)
    #----------------------END----------------------
    
    ground_ret, cam_para_ret = get_ground(scene_type, img, joints_2d, path_vis_folder=path_vis, flag_vis=not close_vis, gt_cam_path=gt_cam_path)

    print('solving done.')
    print(ground_ret)
    print(cam_para_ret)
    
    np.save(os.path.join(path_pre_process, 'ground.npy'), ground_ret)
    np.save(os.path.join(path_pre_process, 'cam_para.npy'), cam_para_ret)
    np.save(os.path.join(path_pre_process, 'scene_shape.npy'), np.array([img.shape[1], img.shape[0]]))

    def vis_joints() :
        img_vis_2d = vis_joint_2d(img, joints_2d)
        save_path_2d = os.path.join(path_fitting_cache, 'vis', scene_type, scene_name, 'img_vis_2d.jpg')
        cv2.imwrite(save_path_2d,img_vis_2d)
    #vis_joints()

