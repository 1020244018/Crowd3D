

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
from lib.ground.loss import get_projection_loss
from lib.ground.solve.initial_value import (
    get_init_ground,
    get_ground_with_f_mapping_to_N
)


from lib.ground.solve.solve_ground_and_cam import argmin_ground


COUNT_CLOSURE = 0

def sovle_ground_only(
        ground_init:np.ndarray,
        xb:np.ndarray, 
        xt:np.ndarray, 
        camera_gt:np.ndarray,
        H_prior:float=1.3,
        max_iter:int=3000,
        learn_rate:float=0.1,
        vis_data_pack:Dict={'flag':False},
        device=torch.device('cpu')
    ):
    '''
    input:
        input are not tensor
    output:
        * ground_ret [4] np.ndarray
        * cam_para_ret [3, 3] np.ndarray
    '''

    print('sovle_ground_only')

    if ground_init[2] > 0 :
        ground_init = -1.0 * ground_init

    #init
    ground_variable = torch.from_numpy(ground_init).to(torch.float32).to(device)
    cam_para_tensor = torch.from_numpy(camera_gt).to(torch.float32).to(device)

    D = ground_variable[3]

    xb_tensor = torch.from_numpy(xb).to(torch.float32).to(device)
    xt_tensor = torch.from_numpy(xt).to(torch.float32).to(device)

    def forward(ground_variable:torch.Tensor, vis_data_pack={'flag': False}) -> (torch.Tensor):
        ground_tensor = ground_variable.clone()
        ground_tensor[3] = D
        ground_normal = ground_tensor[0:3].clone() / torch.norm(ground_tensor[0:3], p=2)

        #decide "depth"
        xyz_bot = uv_to_xyz_via_ground_torch(xb_tensor, ground_tensor, cam_para_tensor)
        #get person
        xyz_top = (xyz_bot.permute(1, 0) + (ground_normal * H_prior)).permute(1, 0)
        #project to image
        uv_top = torch.matmul(cam_para_tensor, xyz_top)
        uv_top = uv_top / uv_top[2, :] 

        loss_vec, loss_mod, loss_pixel = get_projection_loss(
            xb_gt=xb_tensor,
            xt_gt=xt_tensor,
            xt_pred=uv_top,
            vis_data_pack=vis_data_pack
        )

        return loss_vec, loss_mod, loss_pixel

    
    K1 = 0
    K2 = 1
    K3 = 1

    torch.set_printoptions(precision=8, sci_mode=False)

    ground_variable = torch.from_numpy(ground_init).to(torch.float32).to(device)
    ground_variable.requires_grad_(True)
    opt = torch.optim.LBFGS([ground_variable], lr=learn_rate, max_iter=max_iter // 4)
    opt = torch.optim.LBFGS([ground_variable], lr=learn_rate * 0.1, max_iter=max_iter // 2)
    opt = torch.optim.LBFGS([ground_variable], lr=learn_rate * 0.01, max_iter=max_iter)
    
    global COUNT_CLOSURE
    COUNT_CLOSURE = 0

    def closure001() :
        global COUNT_CLOSURE
        COUNT_CLOSURE = COUNT_CLOSURE + 1
        opt.zero_grad()
        loss_vec, loss_mod, loss_pixel = forward(ground_variable)
        loss = loss_vec * 0 + loss_mod * K2 + loss_pixel * K3
        if COUNT_CLOSURE % 100 == 0:
            print(
                'loss and f', loss.detach().item(), 
                loss_mod.detach().item(), loss_pixel.detach().item()
            )
        loss.backward()
        return loss

    opt.step(closure001)
    ground_variable.requires_grad_(False)

    K1_ret = 0
    K2_ret = 0
    K3_ret = 1

    #vis src and target
    vis_data_pack['prefix'] = '%.1f' % cam_para_tensor[0, 0].cpu().item()
    loss_vec, loss_mod, loss_pixel = forward(ground_variable, vis_data_pack=vis_data_pack)
    loss = loss_vec * K1_ret + loss_mod * K2_ret + loss_pixel * K3_ret
    print('opt info(argmin_ground): f and loss %.1f %.4f %.4f %.4f' % (
            float(cam_para_tensor[0, 0].cpu().item()),
            float(loss.detach().item()), 
            float(loss_mod.detach().item()),
            float(loss_pixel.detach().item())
    ))

    ground_ret = ground_variable.cpu().numpy()
    ground_ret[3] = D
    loss_ret = loss.cpu().numpy()
    
    ground_ret = ground_ret / np.linalg.norm(ground_ret[0:3])
    return ground_ret, loss_ret


def solve_ground_with_GT_cam(
        xb:np.ndarray, 
        xt:np.ndarray, 
        image_size:Tuple[int, int],
        camera_gt:np.ndarray,
        H_prior:float=1.3,
        vis_data_pack:Dict={'flag': False},
        device=torch.device('cpu')
    ) -> (Tuple[np.ndarray, np.ndarray]) :

    ground_init, loss_init = get_ground_with_f_mapping_to_N(
        xb=xb, 
        xt=xt, 
        cam_para=camera_gt, 
        vis_data_pack=vis_data_pack
    )

    ground, loss = sovle_ground_only(
        ground_init=ground_init,
        xb=xb,
        xt=xt,
        camera_gt=camera_gt,
        H_prior=H_prior,
        vis_data_pack=vis_data_pack,
        device=device
    )

    return ground

