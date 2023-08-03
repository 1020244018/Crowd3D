

from typing import List, Dict

import numpy as np
import cv2 as cv
import os

import torch
import torch.optim

from lib.ground.solve import (
    argmin_f_search, 
    sovle_ground_opt, 
    get_init_depth_and_ground, 
    manual_set_cam_para_list,
    solve_ground_with_GT_cam
)
from lib.util.transform import filter_by_value_mask


def get_ground(
        name_video:str,
        image:np.ndarray, 
        joint_2d:np.ndarray, 
        path_vis_folder:str,
        scene_mask:np.ndarray or None=None,
        flag_vis:bool=False,
        flag_manual_set_f:bool=False,
        gt_cam_path:str or None=None
    ) :
    '''
    input:
        joint_2d : numpy (person_numcoco_17, 3)
        image : for visualization
        save_path : sava path/name_video/scene_name/
        gt_cam_path : if not None, use gt cam
    output:
        sava_Xb.T : 
        normal : ground normal  list (3,)
    save_file: (for visualization)
        jpg * 2
        obj * 2
    '''
    
    H, W, C = image.shape
    #print('get_ground')
    #print('image shape:', image.shape)

    np.set_printoptions(suppress=True)
    num = joint_2d.shape[0]
    
    xb = np.ones((3, num))  #bot
    xt = np.ones((3, num))  #top

    #put 0, 0 in the center of image
    for i in range(num) :
        for j in [0, 1] :
            if j == 0 :
                xb[j, i] = (joint_2d[i][15][j] + joint_2d[i][16][j]) / 2
                xt[j, i] = (joint_2d[i][5][j] + joint_2d[i][6][j]) / 2
            elif j == 1 :
                xb[j, i] = (joint_2d[i][15][j] + joint_2d[i][16][j]) / 2
                xt[j, i] = (joint_2d[i][5][j] + joint_2d[i][6][j]) / 2
            else :
                raise(NotImplementedError(j))

    if scene_mask is not None:
        xb_mask = filter_by_value_mask(xb, scene_mask)
        xb = xb[:, xb_mask]
        xt = xt[:, xb_mask]

    vis_data_pack = {'flag': False, 'path':path_vis_folder}
    #vis ready
    if flag_vis :
        print('vis enabled.')
        ratio_aspect = W / H
        ratio_scale_vis = 1
        image_vis = cv.resize(image, (int(W*ratio_scale_vis), int(H*ratio_scale_vis)), interpolation=cv.INTER_LINEAR)
        vis_data_pack = {
            'flag': True,
            'image': image_vis,
            'scale': ratio_scale_vis,
            'path': path_vis_folder
        }

    device = torch.device('cuda')

    if not flag_manual_set_f:
        #---------new()---------
        if gt_cam_path is None:
            '''
            solve gt cam and ground
            '''
            ground_ret, cam_para_ret = argmin_f_search(
                xb=xb,
                xt=xt, 
                image_size=image.shape[0:2],
                H_prior=1.40,
                init_fov=[20, 135],
                vis_data_pack=vis_data_pack,
                device=device
            )

            print('after argmin_f_search.')
            print(ground_ret)

            ground_ret = solve_ground_with_GT_cam(
                xb=xb, xt=xt, image_size=image.shape[0:2],
                H_prior=1.40,
                camera_gt=cam_para_ret,
                vis_data_pack=vis_data_pack,
                device=device
            )
        else:
            '''
            solve ground with gt cam given
            '''
            camera_gt = np.load(gt_cam_path)
            assert type(camera_gt) and camera_gt.shape == (3,3), camera_gt
            ground_ret = solve_ground_with_GT_cam(
                xb=xb, xt=xt, image_size=image.shape[0:2],
                H_prior=1.35,
                camera_gt=camera_gt,
                vis_data_pack=vis_data_pack,
                device=device
            )
            cam_para_ret = camera_gt
            
    else:
        #---------old()---------
        #only the result with last cam_para in the cam_para_list
        #will be saved
        #but the losses of all cam_para will be print
        cam_para_list = manual_set_cam_para_list(W, H, name_video)
        for cam_para in cam_para_list:
            #---------BEGIN:get init depth---------
            ground_init = get_init_depth_and_ground(xb, xt, cam_para, vis_data_pack=vis_data_pack)
            print('ground_init', ground_init)
            #---------END:get init depth---------
            #---------BEGIN:solve ground---------
            ground_ret, cam_para_ret = sovle_ground_opt(xb, xt, ground_init, cam_para, H_prior=1.32, vis_data_pack=vis_data_pack, device=device)
            #---------END:solve ground---------

    return ground_ret, cam_para_ret

