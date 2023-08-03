
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

CLOUSER_MIN_LOSS = 9999999
CLOUSER_BEST_PARAM = None

def argmin_ground(
        f:float,
        known_data_pack:Dict,
        ground_init:np.ndarray,
        cam_para:np.ndarray,
        max_iter:int=500,
        learn_rate:float=0.01,
        opt_alg:torch.optim.Optimizer=torch.optim.LBFGS,
        vis_data_pack:Dict={'flag':False}
    ):
    '''
    input:
        input are not tensor
    output:
        * ground_ret [4] np.ndarray
        * cam_para_ret [3, 3] np.ndarray
    '''
    CLOUSER_MIN_LOSS = 9999999
    CLOUSER_BEST_PARAM = None

    xb = known_data_pack['xb']
    xt = known_data_pack['xt']
    image_size = known_data_pack['image_size']
    H_prior = known_data_pack['H_prior']
    device = known_data_pack['device']

    if ground_init[2] > 0 :
        ground_init = -1.0 * ground_init

    #init
    ground_variable = torch.from_numpy(ground_init).to(torch.float32).to(device)
    cam_para_tensor = torch.from_numpy(cam_para).to(torch.float32).to(device)
    cam_para_tensor[0, 0] = f
    cam_para_tensor[1, 1] = f

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

    '''
    vis_data_pack['prefix'] = '%.1f_before' % cam_para_tensor[0, 0].cpu().item()
    loss_vec, loss_mod, loss_pixel = forward(ground_variable, vis_data_pack=vis_data_pack)
    loss = loss_vec * K1 + loss_mod * K2 + loss_pixel * K3
    print('opt info(argmin_ground) before: f and loss %.1f %.4f %.4f %.4f %.4f' % (
            float(cam_para_tensor[0, 0].cpu().item()),
            float(loss.detach().item()), 
            float(loss_vec.detach().item()), 
            float(loss_mod.detach().item()),
            float(loss_pixel.detach().item())
    ))
    '''
    def opt_LBFGS():
        def closure001() :
            global CLOUSER_MIN_LOSS
            global CLOUSER_BEST_PARAM
            opt.zero_grad()
            loss_vec, loss_mod, loss_pixel = forward(ground_variable)
            loss = loss_vec * 0 + loss_mod * K2 + loss_pixel * K3
            print(
                'loss and f', loss.detach().item(), 
                #loss_vec.detach().item(), 
                loss_mod.detach().item(), loss_pixel.detach().item()
            )
            if loss < CLOUSER_MIN_LOSS:
                CLOUSER_MIN_LOSS = loss
                CLOUSER_BEST_PARAM = ground_variable.detach().clone()
            loss.backward()
            return loss

        def closure011() :
            global CLOUSER_MIN_LOSS
            global CLOUSER_BEST_PARAM
            opt.zero_grad()
            loss_vec, loss_mod, loss_pixel = forward(ground_variable)
            loss = loss_vec * 0 + loss_mod * K2 + loss_pixel * K3
            print(
                'loss and f', loss.detach().item(), 
                #loss_vec.detach().item(), 
                loss_mod.detach().item(), loss_pixel.detach().item()
            )
            if loss < CLOUSER_MIN_LOSS:
                CLOUSER_MIN_LOSS = loss
                CLOUSER_BEST_PARAM = ground_variable.detach().clone()
            loss.backward()
            return loss

        ground_variable = torch.from_numpy(ground_init).to(torch.float32).to(device)
        ground_variable.requires_grad_(True)
        opt = torch.optim.LBFGS([ground_variable], lr=learn_rate, max_iter=100)
        opt.step(closure001)

        #ground_variable = torch.from_numpy(ground_init).to(torch.float32).to(device)
        #ground_variable.requires_grad_(True)
        opt = torch.optim.LBFGS([ground_variable], lr=learn_rate, max_iter=max_iter)
        opt.step(closure011)
        ground_variable.requires_grad_(False)
        return ground_variable

    def opt_SGD():
        ground_variable.requires_grad_(True)
        opt = torch.optim.SGD([ground_variable], lr=learn_rate)
        for i in range(max_iter):
            opt.zero_grad()
            loss_vec, loss_mod, loss_pixel = forward(ground_variable)
            loss = loss_vec * K1 + loss_mod * K2 + loss_pixel * K3
            print('f and loss %.1f %.4f %.4f %.4f %.4f' % (
                    float(cam_para_tensor[0, 0].cpu().item()),
                    float(loss.detach().item()), 
                    #float(loss_vec.detach().item()), 
                    float(loss_mod.detach().item()),
                    float(loss_pixel.detach().item())
            ))
            loss.backward()
            opt.step()
        ground_variable.requires_grad_(False)
        return ground_variable

    if opt_alg == torch.optim.LBFGS:
        ground_variable = opt_LBFGS()
        loss_vec, loss_mod, loss_pixel = forward(ground_variable)
        loss = loss_vec * K1 + loss_mod * K2 + loss_pixel * K3
        if False:
            if loss > CLOUSER_MIN_LOSS:
                print('Unstable LBFGS results is detected, use minial error param during the opt.')
                print('CLOUSER_MIN_LOSS, CLOUSER_BEST_PARAM', (CLOUSER_MIN_LOSS, CLOUSER_BEST_PARAM))
                loss = CLOUSER_MIN_LOSS
                ground_variable = CLOUSER_BEST_PARAM

    elif opt_alg == torch.optim.SGD:
        raise(NotImplementedError())
        opt_SGD()
    elif opt_alg == 'Mixed':
        raise(NotImplementedError())
        ground_variable = opt_LBFGS()
        loss_vec, loss_mod, loss_pixel = forward(ground_variable)
        loss = loss_vec * K1 + loss_mod * K2 + loss_pixel * K3
        if loss > CLOUSER_MIN_LOSS:
            print('Mixed alg working... because unstable LBFGS results is detected')
            ground_variable = opt_SGD()
        else:
            print('Mixed alg is not active because LBFGS is good enough.')
    else:
        raise(NotImplementedError())

    #vis src and target
    forward(ground_variable)
    vis_data_pack['prefix'] = '%.1f_argmin_ground' % cam_para_tensor[0, 0].cpu().item()
    loss_vec, loss_mod, loss_pixel = forward(ground_variable, vis_data_pack=vis_data_pack)
    loss = loss_vec * K1 + loss_mod * K2 + loss_pixel * K3
    print('opt info(argmin_ground): f and loss %.1f %.4f %.4f %.4f' % (
            float(cam_para_tensor[0, 0].cpu().item()),
            float(loss.detach().item()), 
            #float(loss_vec.detach().item()), 
            float(loss_mod.detach().item()),
            float(loss_pixel.detach().item())
    ))

    ground_ret = ground_variable.cpu().numpy()
    loss_ret = loss.cpu().numpy()
    return ground_ret, loss_ret


