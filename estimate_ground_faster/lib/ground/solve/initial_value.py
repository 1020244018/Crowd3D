

from typing import List, Dict, Tuple

import numpy as np
import cv2 as cv
import os

import torch
import torch.optim

from lib.vis.vis_ground import vis_point_with_info, vis_ground_grid
from lib.solve.linear import solve_ground_from_xyz
from lib.ground.solve.solve_ground_d import solve_D_opt
from lib.util.util_py import generate_info

def get_KN(
    xb:np.ndarray, 
    xt:np.ndarray,
    flag_ret_A:bool=False
    ):
    assert xb.shape == xt.shape, (xb.shape, xt.shape)
    assert xb.shape[0] == 3, xb.shape
    num = xb.shape[1]
    A = np.zeros((num, 3))
    for i in range(num):
        A[i, :] = np.cross(xb[:, i], xt[:, i])
    ____, s, v = np.linalg.svd(A)
    #vanishing point is KN
    KN = v[np.argmin(s), :]
    KN = KN / KN[2]
    if flag_ret_A:
        return KN, A
    else:
        return KN

def get_KN_with_filter(
    xb:np.ndarray, 
    xt:np.ndarray
    ):
    assert xb.shape == xt.shape, (xb.shape, xt.shape)
    assert xb.shape[0] == 3, xb.shape
    

    def filter_by_value(xb, xt, values, max):
        xb_ret = []
        xt_ret = []
        for i, value in enumerate(values):
            if value < max:
                xb_ret.append(xb[:, i])
                xt_ret.append(xt[:, i])
        return np.array(xb_ret).T, np.array(xt_ret).T

    def loop_filter(_xb:np.ndarray, _xt:np.ndarray, _prop:float=0.2):
        assert _xb.shape == _xt.shape, (_xb.shape, _xt.shape)
        assert _xb.shape[0] == 3, _xb.shape
        _num = _xb.shape[1]
        _KN, _A = get_KN(_xb, _xt, flag_ret_A=True)
        _values = np.abs(np.matmul(_A, _KN))
        _values_sorted = sorted(_values)
        _xb2, _xt2 = filter_by_value(_xb, _xt, _values, _values_sorted[int(_num*_prop)])
        return _xb2, _xt2, _KN

    xb_tmp = xb.copy()
    xt_tmp = xt.copy()
        
    xb_tmp, xt_tmp, KN_RET = loop_filter(xb_tmp, xt_tmp, _prop=0.75)

    return KN_RET

def get_ground_with_f_mapping_to_N(
    xb:np.ndarray, 
    xt:np.ndarray, 
    cam_para:np.ndarray,
    H_prior:float=1.3,
    D_init:float=10.0,
    vis_data_pack:Dict={'flag': False}
    ) -> (Tuple[np.ndarray, np.ndarray]):
    '''
    output a init ground by considering the depth of Xb and Xt are equal
    input:
        xt [3, N] np.ndarray
        xb [3, N] np.ndarray
        cam_para [3, 3] np.ndarray: camera instinct matrix
    '''
    assert xb.shape == xt.shape, (xb.shape, xt.shape)
    assert xb.shape[0] == 3, xb.shape
    assert cam_para.shape == (3, 3), cam_para

    num = xb.shape[1]

    if vis_data_pack['flag'] :
        assert 'image' in vis_data_pack.keys()
        assert 'scale' in vis_data_pack.keys()
        assert 'path' in vis_data_pack.keys()
        assert os.path.exists(vis_data_pack['path']), vis_data_pack['path']

    KN = get_KN_with_filter(xb, xt)

    ground_normal = np.matmul(np.linalg.inv(cam_para), KN)
    ground_normal = ground_normal / np.linalg.norm(ground_normal, ord=2)

    ground, loss = solve_D_opt(
        xb=xb,
        xt=xt,
        ground_normal=ground_normal,
        cam_para=cam_para,
        H_prior=H_prior,
        D_init=D_init,
        vis_data_pack=vis_data_pack
    )
    
    #print('init ground:', ground)
    if vis_data_pack['flag'] :
        assert 'image' in vis_data_pack.keys()
        assert 'scale' in vis_data_pack.keys()
        assert 'path' in vis_data_pack.keys()
        assert os.path.exists(vis_data_pack['path']), vis_data_pack['path']
    if vis_data_pack['flag'] :
        #vis ground
        vis_ground_grid(
            vis_data_pack=vis_data_pack,
            grid_direction='parallel',
            point_uv1t=xb,
            ground=ground,
            cam_para=cam_para
        )
    return ground, loss


def get_init_ground(
    xb:np.ndarray, 
    xt:np.ndarray, 
    cam_para:np.ndarray,
    H_prior:float=1.3,
    vis_data_pack:Dict={'flag': False}
    ) :
    '''
    DESPERATED
    output a init ground by considering the depth of Xb and Xt are equal
    input:
        xt [3, N] np.ndarray
        xb [3, N] np.ndarray
        cam_para [3, 3] np.ndarray: camera instinct matrix
    '''

    assert xb.shape == xt.shape, (xb.shape, xt.shape)
    assert xb.shape[0] == 3, xb.shape
    assert cam_para.shape == (3, 3), cam_para

    num = xb.shape[1]
    Xt = np.matmul(np.linalg.inv(cam_para), xt)
    Xb = np.matmul(np.linalg.inv(cam_para), xb)
    H = np.sum((Xt - Xb) ** 2, axis=0) ** 0.5
    #print(H)
    Z_init = 1 * H_prior / H
    #print(Depth)
    #3d point in camera coor
    point_3d_cam = np.matmul(np.linalg.inv(cam_para), xb * Z_init)
    assert point_3d_cam.shape[1] == num
    ground = solve_ground_from_xyz(point_3d_cam.T)

    #print('init ground:', ground)
    if vis_data_pack['flag'] :
        assert 'image' in vis_data_pack.keys()
        assert 'scale' in vis_data_pack.keys()
        assert 'path' in vis_data_pack.keys()
        assert os.path.exists(vis_data_pack['path']), vis_data_pack['path']
    if vis_data_pack['flag'] :
        #vis init H and D
        info=generate_info(['H', 'Z'], [H, Z_init])
        vis_point_with_info(
            vis_data_pack,
            point_uv1t=xt,
            info=info
        )
    if vis_data_pack['flag'] :
        #vis ground
        vis_ground_grid(
            vis_data_pack=vis_data_pack,
            grid_direction='parallel',
            point_uv1t=xb,
            ground=ground,
            cam_para=cam_para
        )
    return ground

def manual_set_cam_para_list(W:int, H:int, name_video:str) -> (List[np.ndarray]) :
    '''
    in use, used to be discarded
    '''
    if '02' in name_video:
        list_f = list(range(18500, 22000, 250))
        #list_f = [19500 - 50, 19500, 19500 + 50]
        #list_f = [19500]
    elif '05' in name_video :
        list_f = list(range(78000, 84000, 1000))
        #list_f = [81000 - 250, 81000, 81000 + 250]
        #list_f = [81000]
    elif '07' in name_video :
        list_f = list(range(17000, 21000, 250))
        #list_f = [17800 - 50, 17800, 17800 + 50]
        #list_f = [17850] #[17000]
    elif '10' in name_video :
        #list_f = list(range(18000, 24000, 250))
        #list_f = [23900 - 50, 23900, 23900 + 50]
        list_f = [22250]
    else :
        raise(NotImplementedError(name_video))

    list_cam_para = []
    for f in list_f:
        cam_in = np.array(
            [[f, 0, W/2],
            [0, f, H/2],
            [0, 0, 1]], dtype=np.float32
        )
        list_cam_para.append(cam_in)

    '''
    DESPERATED
    if '02' in name_video or '07' in name_video or '10' in name_video:
        f = 16000 * W / 26753.0
        cam_in = np.array(
            [[f, 0, W/2],
            [0, f, H/2],
            [0, 0, 1]], dtype=np.float32
        )
    elif '05' in name_video :
        f = 60000 * W / 31746.0
        cam_in = np.array(
            [[f, 0, W/2],
            [0, f, H/2],
            [0, 0, 1]], dtype=np.float32
        )
    elif '11' in name_video:
        #f = 25000 * W / 26583.0
        f = 18500 * W / 26583.0
        cam_in = np.array(
            [[f, 0, W/2],
            [0, f, H/2],
            [0, 0, 1]], dtype=np.float32
        )
    else :
        raise(NotImplementedError(name_video))
    '''

    return list_cam_para


def get_init_depth_and_ground(
    xb:np.ndarray, 
    xt:np.ndarray, 
    cam_para:np.ndarray,
    H_prior:float=1.5,
    vis_data_pack:Dict={'flag': False}
    ) :
    '''
    discarded
    input:
        xt [3, N] np.ndarray
        xb [3, N] np.ndarray
        cam_para [3, 3] np.ndarray: camera instinct matrix
    '''

    assert xb.shape == xt.shape, (xb.shape, xt.shape)
    assert xb.shape[0] == 3, xb.shape
    assert cam_para.shape == (3, 3), cam_para

    num = xb.shape[1]

    if vis_data_pack['flag'] :
        assert 'image' in vis_data_pack.keys()
        assert 'scale' in vis_data_pack.keys()
        assert 'path' in vis_data_pack.keys()
        assert os.path.exists(vis_data_pack['path']), vis_data_pack['path']

    A = np.zeros((num, 3))
    for i in range(num):
        A[i, :] = np.cross(xb[:, i], xt[:, i])
    ____, s, v = np.linalg.svd(A)
    #vanishing point is KN
    KN = v[np.argmin(s), :]
    KN = KN / KN[2]
    ground_normal = np.matmul(np.linalg.inv(cam_para), KN)
    ground_normal = ground_normal / np.linalg.norm(ground_normal, ord=2)

    D_tmp = 5.0
    ground, loss = solve_D_opt(
        xb=xb,
        xt=xt,
        ground_normal=ground_normal,
        cam_para=cam_para,
        H_prior=H_prior,
        vis_data_pack=vis_data_pack
    )
    return ground
    #exit()

    #num = xb.shape[1]

    Xt = np.matmul(np.linalg.inv(cam_para), xt)
    Xb = np.matmul(np.linalg.inv(cam_para), xb)

    H = np.sum((Xt - Xb) ** 2, axis=0) ** 0.5
    #print(H)
    Z_init = 1 * H_prior / H
    #print(Depth)

    #3d point in camera coor
    point_3d_cam = np.matmul(np.linalg.inv(cam_para), xb * Z_init)

    assert point_3d_cam.shape[1] == num

    ground = solve_ground_from_xyz(point_3d_cam.T)

    #print('init ground:', ground)

    if False: #vis_data_pack['flag'] :
        #vis init H and D
        info=generate_info(['H', 'Z'], [H, Z_init])
        vis_point_with_info(
            vis_data_pack,
            point_uv1t=xt,
            info=info
        )

    if vis_data_pack['flag'] :
        #vis ground
        vis_ground_grid(
            vis_data_pack=vis_data_pack,
            grid_direction='parallel',
            point_uv1t=xb,
            ground=ground,
            cam_para=cam_para
        )

    return ground