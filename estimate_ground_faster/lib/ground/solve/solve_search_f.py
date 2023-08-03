

from typing import List, Dict, Tuple

import numpy as np
import cv2 as cv
import os
import copy

import torch
import torch.optim

from lib.vis.vis_ground import vis_ground_grid

from lib.ground.util import (
    uv_to_xyz_via_ground,
    uv_to_xyz_via_ground_torch,
    get_cam_in_matrix,
    lpfc_PANDA_shenzhenbei
)
from lib.util.search_util import (
    get_next_range_with_unstable_allowance,
    get_list_fov
)
# from lib.ground.loss import get_projection_loss
from lib.ground.solve.initial_value import (
    get_init_ground,
    get_ground_with_f_mapping_to_N
)


from lib.ground.solve.solve_ground_and_cam import argmin_ground


def search_f_once(
    list_fov:List[float],
    max_iter:int,
    learn_rate:float,
    opt_alg:torch.optim.Optimizer,
    known_data_pack:Dict,
    vis_data_pack:Dict,
    flag_opt_GN:bool=False
    ) -> (Tuple[np.ndarray, np.ndarray]) :

    xb = known_data_pack['xb']
    xt = known_data_pack['xt']
    image_size = known_data_pack['image_size']
    #H_prior = known_data_pack['H_prior']
    #device = known_data_pack['device']

    H = image_size[0]
    W = image_size[1]

    list_loss = []
    list_cam_para = []
    list_ground = []
    list_F = []

    for fov in list_fov:

        f = 0.5/np.tan(fov*np.pi/180/2.0)
        F = f*W

        #print('F', F)

        cam_para = get_cam_in_matrix(F, W=W, H=H)

        #vis_data_pack['flag'] = True
        ground_init, loss_init = get_ground_with_f_mapping_to_N(
            xb=xb, 
            xt=xt, 
            cam_para=cam_para, 
            vis_data_pack=vis_data_pack
        )

        #flag_opt_GN = True

        if not flag_opt_GN:
            list_ground.append(ground_init)
            list_loss.append(loss_init)
            list_cam_para.append(cam_para)
            list_F.append(F)
            continue
        else:
            #vis_data_pack['flag'] = True
            ground, loss = argmin_ground(
                f=F,
                known_data_pack=known_data_pack,
                ground_init=ground_init,
                cam_para=cam_para,
                max_iter=max_iter,
                learn_rate=learn_rate,
                opt_alg=opt_alg,
                vis_data_pack=vis_data_pack
            )
            
            list_ground.append(ground)
            list_loss.append(loss)
            list_cam_para.append(cam_para)
            list_F.append(F)
    #print(list_loss)
    #print(list_F)
    #exit()
    return list_loss, list_ground, list_cam_para


def argmin_f_search(
        xb:np.ndarray, 
        xt:np.ndarray, 
        image_size:Tuple[int, int],
        H_prior:float=1.3,
        init_fov:List[float]=[50, 110],
        vis_data_pack:Dict={'flag': False},
        device=torch.device('cpu'),
        flag_no_strict_condition:bool=True,
    ) -> (Tuple[np.ndarray, np.ndarray]) :
    '''
        W/2
        .----.
        |   /
        |F /
        | /
        |/
        .
        FoV = 2*arctan(W/2/f)
        F = W * (1/2/tan(FoV/2))
        set f = 1/2/tan(FoV/2)

    input:
    
    output:
        * ground_ret [4] np.ndarray
        * cam_para_ret [3, 3] np.ndarray
    '''
    H = image_size[0]
    W = image_size[1]

    known_data_pack = {
        'xb':xb,
        'xt':xt,
        'image_size':image_size,
        'H_prior':H_prior,
        'device':device
    }

    '''
      
    '''
    #begin
    range_fov = init_fov
    list_fov = get_list_fov(range_fov, 7)

    #range_fov = [39.1, 39.1]
    #list_fov = get_list_fov(range_fov, 5)

    learn_rate = 0.025
    max_iter = 125

    #first search
    list_loss, list_ground, list_cam_para = search_f_once(
        list_fov,
        learn_rate=learn_rate,
        max_iter=max_iter,
        opt_alg=torch.optim.LBFGS,
        known_data_pack=known_data_pack,
        vis_data_pack=vis_data_pack
    )

    list_f = []
    for fov in list_fov:
        f = 0.5/np.tan(fov*np.pi/180/2.0)
        F = f*W
        list_f.append(F)
  
    #find the min index p3
    #and 5 point in total p1, p2, p3, p4, p5
    #check all the 5 point 
    #then select p2-p4
    #start to search by trisection method
    count=0
    while True:
        index_range = get_next_range_with_unstable_allowance(list_loss)

        if flag_no_strict_condition:
            '''
            loose condition
            if the loss is not convex, end search and output
            '''
            if index_range is None:
                index_min = np.argmin(np.array(list_loss))
                ground_ret = list_ground[index_min]
                cam_para_ret = list_cam_para[index_min]
                break
        else:
            '''
            strict condition
            if the loss is not convex, raise Exception
            '''
            if index_range is None:
                raise(Exception('the loss is not convex. Please refer to the same warning.'))

        range_fov = [list_fov[index_range[0]], list_fov[index_range[1]]]
        epsilon_1 = range_fov[1] - range_fov[0]
        epsilon_2 = np.min(np.array(list_loss))

        count+=1
        # print('This round searching info:')
        # print('list_f', list_f)
        # print('list_loss', np.array(list_loss))
        # print('epsilon_1', epsilon_1)
        # print('epsilon_2', epsilon_2)
        print('Round %d: fov_delta %.2f, min_loss %.5f'  %(count, epsilon_1, epsilon_2))
        
        if epsilon_1 < 1:
            list_fov_finetune = get_list_fov(range_fov, 3)

            '''
            list_loss_gn, list_ground_gn, list_cam_para_gn = search_f_once(
                list_fov_finetune,
                learn_rate=learn_rate,
                max_iter=max_iter,
                opt_alg=torch.optim.LBFGS,
                known_data_pack=known_data_pack,
                vis_data_pack=vis_data_pack,
                flag_opt_GN=True
            )

            list_loss = list_loss + list_loss_gn
            list_loss = list_ground + list_ground_gn
            list_loss = list_cam_para + list_cam_para_gn
            '''

            index_min = np.argmin(np.array(list_loss))
            ground_ret = list_ground[index_min]
            cam_para_ret = list_cam_para[index_min]
            break
        else:
            pass

        list_fov = get_list_fov(range_fov, 10)

        #for visualization
        list_f = []
        for fov in list_fov:
            f = 0.5/np.tan(fov*np.pi/180/2.0)
            F = f*W
            list_f.append(F)

        list_loss, list_ground, list_cam_para = search_f_once(
            list_fov,
            learn_rate=learn_rate,
            max_iter=max_iter,
            opt_alg=torch.optim.LBFGS,
            known_data_pack=known_data_pack,
            vis_data_pack=vis_data_pack
        )

        
        
    return ground_ret, cam_para_ret



    





