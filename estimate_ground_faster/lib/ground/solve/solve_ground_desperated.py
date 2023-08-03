

from typing import List, Dict, Tuple

import numpy as np
import cv2 as cv
import os

import torch
import torch.optim

from lib.vis.vis_ground import vis_ground_grid

from lib.ground.util import (
    uv_to_xyz_via_ground_torch
)
from lib.ground.loss import get_projection_loss


def sovle_ground_opt(
    xb:np.ndarray, 
    xt:np.ndarray, 
    ground_init:np.ndarray,
    cam_para:np.ndarray,
    H_prior:float=1.5,
    vis_data_pack:Dict={'flag': False},
    device=torch.device('cpu')
    ) -> (Tuple[np.ndarray, np.ndarray]) :
    '''
    input:
    
    output:
        * ground_ret [4] np.ndarray
        * cam_para_ret [3, 3] np.ndarray
    '''

    if ground_init[2] > 0 :
        ground_init = -1.0 * ground_init

    #init
    ground_variable = torch.from_numpy(ground_init).to(torch.float32).to(device)
    f_variable = torch.ones((1)).to(torch.float32).to(device) * float(cam_para[0, 0])
    cx = float(cam_para[0, 2])
    cy = float(cam_para[1, 2])

    D = ground_variable[3]

    xb_tensor = torch.from_numpy(xb).to(torch.float32).to(device)
    xt_tensor = torch.from_numpy(xt).to(torch.float32).to(device)


    def forward(ground_variable:torch.Tensor, f:torch.Tensor, vis_data_pack={'flag': False}) -> (torch.Tensor):
        ground_tensor = ground_variable.clone()
        ground_tensor[3] = D

        cam_para_tensor = torch.zeros((3, 3), dtype=torch.float32, device=device)
        cam_para_tensor[0, 0] = f
        cam_para_tensor[1, 1] = f
        cam_para_tensor[2, 2] = 1
        cam_para_tensor[0, 2] = cx
        cam_para_tensor[1, 2] = cy

        ground_normal = ground_tensor[0:3].clone() / torch.norm(ground_tensor[0:3], p=2)
        #print(ground_normal)
        #if ground_normal[1] < 0 :
        #    ground_normal = - ground_normal

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

        #print(loss_pixel, f)

        return loss_vec, loss_mod, loss_pixel

    K1 = 0
    K2 = 0
    K3 = 1

    def closure() :
        opt.zero_grad()
        loss_vec, loss_mod, loss_pixel = forward(ground_variable, f_variable)
        loss = K1 * loss_vec + K2 * loss_mod + K3 * loss_pixel
        #print(loss, f_variable)
        loss.backward()
        return loss

    ground_variable.requires_grad_(True)
    #f_variable.requires_grad_(True)
    opt = torch.optim.LBFGS([ground_variable], lr=0.1, max_iter=2000)#, line_search_fn='strong_wolfe')
    opt.step(closure)
    ground_variable.requires_grad_(False)
    #f_variable.requires_grad_(False)

    '''
    ground_variable.requires_grad_(True)
    f_variable.requires_grad_(True)
    opt = torch.optim.SGD([
        {'params': ground_variable},
        {'params': f_variable, 'lr': 1}
    ], lr=0.01, momentum=0.9)
    for i in range(500):
        opt.zero_grad()
        loss_vec, loss_mod, loss_pixel = forward(ground_variable, f_variable)
        loss = K1 * loss_vec + K2 * loss_mod + K3 * loss_pixel
        print(loss, f_variable)
        loss.backward()
        opt.step()
    ground_variable.requires_grad_(False)
    f_variable.requires_grad_(False)
    '''
    

    #vis src and target
    loss_vec, loss_mod, loss_pixel = forward(ground_variable, f_variable, vis_data_pack=vis_data_pack)
    loss = K1 * loss_vec + K2 * loss_mod + K3 * loss_pixel
    print('sovle_ground_opt, after opt, loss and f %.5f, %.5f' % (loss.cpu().item(), f_variable.cpu().item()))

    ground_ret = ground_variable.cpu().numpy()
 
    if vis_data_pack['flag'] :
        #vis ground
        vis_ground_grid(
            vis_data_pack=vis_data_pack,
            grid_direction='parallel',
            ground=ground_variable.cpu().numpy(),
            cam_para=cam_para,
            name_save='after_opt'
        )

    cam_para_tensor = torch.zeros((3, 3), dtype=torch.float32, device=device)
    cam_para_tensor[0, 0] = f_variable.cpu()
    cam_para_tensor[1, 1] = f_variable.cpu()
    cam_para_tensor[2, 2] = 1
    cam_para_tensor[0, 2] = cx
    cam_para_tensor[1, 2] = cy
    cam_para_ret = cam_para_tensor.cpu().numpy()

    return ground_ret, cam_para_ret


