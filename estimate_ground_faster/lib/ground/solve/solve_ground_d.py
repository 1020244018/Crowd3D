

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

def solve_D_opt(
        xb:np.ndarray, 
        xt:np.ndarray, 
        ground_normal:np.ndarray,
        cam_para:np.ndarray,
        H_prior:float=1.5,
        D_init:float=10.0,
        vis_data_pack:Dict={'flag': False},
        device=torch.device('cpu')
    ) -> (Tuple[np.ndarray, np.ndarray]):

    if ground_normal[2] > 0 :
        ground_normal = -1.0 * ground_normal
    ground_normal_tensor= torch.from_numpy(ground_normal).to(torch.float32).to(device)
    cam_para_tensor = torch.from_numpy(cam_para).to(torch.float32).to(device)
    xb_tensor = torch.from_numpy(xb).to(torch.float32).to(device)
    xt_tensor = torch.from_numpy(xt).to(torch.float32).to(device)
    D = torch.ones((1)).to(torch.float32).to(device) * D_init

    def forward(D:torch.Tensor, vis_data_pack={'flag': False}) -> (torch.Tensor):
        ground_tensor = torch.zeros((4)).to(torch.float32).to(device)
        ground_tensor[0:3] = ground_normal_tensor
        ground_tensor[3] = D

        #decide "depth"
        xyz_bot = uv_to_xyz_via_ground_torch(xb_tensor, ground_tensor, cam_para_tensor)
        #get person
        xyz_top = (xyz_bot.permute(1, 0) + (ground_normal_tensor * H_prior)).permute(1, 0)
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

    def closure() :
        opt.zero_grad()
        loss_vec, loss_mod, loss_pixel = forward(D)
        loss = K1 * loss_vec + K2 * loss_mod + K3 * loss_pixel
        #print(loss, D)
        loss.backward()
        return loss

    D.requires_grad_(True)
    opt = torch.optim.LBFGS([D], lr=1, max_iter=50)
    opt.step(closure)
    D.requires_grad_(False)

    D.requires_grad_(True)
    opt = torch.optim.LBFGS([D], lr=0.1, max_iter=100)
    opt.step(closure)
    D.requires_grad_(False)
    
    vis_data_pack['prefix'] = str(int(cam_para[0, 0]))
    #vis src and target

    K1_ret = 0
    K2_ret = 0
    K3_ret = 1
    loss_vec, loss_mod, loss_pixel = forward(D, vis_data_pack=vis_data_pack)
    loss_ret = K1_ret * loss_vec + K2_ret * loss_mod + K3_ret * loss_pixel
    #print('after opt, loss and D %.5f, %.5f' % (loss.cpu().item(), D.cpu().item()))

    ground_tensor = torch.zeros((4)).to(torch.float32).to(device)
    ground_tensor[0:3] = ground_normal_tensor
    ground_tensor[3] = D
 
    if vis_data_pack['flag'] :
        #vis ground
        vis_ground_grid(
            vis_data_pack=vis_data_pack,
            grid_direction='parallel',
            ground=ground_tensor.cpu().numpy(),
            cam_para=cam_para
        )

    return ground_tensor.cpu().numpy(), loss_ret.cpu().numpy()

