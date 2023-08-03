

from typing import List, Dict, Tuple
import enum
import numpy as np
import cv2 as cv
import torch

class Data_Shape(enum.Enum):
    homo_2d = 0 #[3, N]

def filter_by_value_mask(x:np.ndarray, valid_mask:np.ndarray, data_shape:Data_Shape=Data_Shape.homo_2d):
    '''
    keep x if valid_mask[int(x)] is true
    '''
    if data_shape == Data_Shape.homo_2d:
        #x is [3, N]
        assert len(x.shape) == 2
        assert x.shape[0] == 3
        N = x.shape[1]

        #mask is [W, H]
        assert len(valid_mask.shape) == 2

        list_mask = []
        for i in range(N):
            value = x[:, i]
            assert value[2] == 1, value
            list_mask.append(valid_mask[int(value[0]), int(value[1])])
    else:
        raise(NotImplementedError())
    return np.array(list_mask, dtype=np.bool8)

def filter_by_value_mask_tuple(
        x:Tuple[np.ndarray], 
        valid_mask:np.ndarray, 
        data_shape:Data_Shape=Data_Shape.homo_2d
    ) -> (Tuple[np.ndarray]):
    '''
    keep x if valid_mask[int(x)] is true
    '''
    if data_shape == Data_Shape.homo_2d:
        assert len(x[0].shape) == 2
        N = x[0].shape[1]
        for elements in x:        
            #elements of x is [3, N]
            assert len(elements.shape) == 2
            assert elements.shape[0] == 3
            assert elements.shape[1] == N

        #mask is [W, H]
        assert len(valid_mask.shape) == 2

        list_ret = []
        for elements in x:    
            list_ret.append([])

        
        for i in range(N):
            flag_valid = True
            for elements in x:
                value = elements[:, i]
                assert value[2] == 1, value
                if not valid_mask[int(value[0]), int(value[1])]:
                    flag_valid = False
            if flag_valid:
                for i_e, ____ in enumerate(x):
                    list_ret[i_e].append(value)

        for i_e, ____ in enumerate(x):
            list_ret[i_e] = np.array(list_ret[i_e], dtype=x[0].dtype).T
    else:
        raise(NotImplementedError())
    return tuple(list_ret)

        


def vector_to_euler_angle(vector:torch.Tensor) :
    '''
    use vector[0] vector[1] vector[2] to access
    vector's mod is 1
    '''
    x = vector[0]
    y = vector[1]
    z = vector[2]
    alpha = torch.arctan(y/x)
    #beta = torch.
    raise(NotImplementedError())

def euler_angle_to_R(angles:torch.Tensor) :
    '''
    use angles[0] angles[1] angles[2] to access
    in radian
    '''
    raise(NotImplementedError())

def uv_to_uv1T(uv:np.ndarray):
    '''
    @param:
        * uv [n, 2]
    @return:
        * uv1T [3, n]
    '''
    assert len(uv.shape) == 2
    assert uv.shape[1] == 2
    n = uv.shape[0]
    uv1T = np.row_stack([uv.T, np.ones((n))])
    return uv1T


def convert_point_vector(point:np.ndarray, CONVERT:str, para:Dict or None=None) :
    if CONVERT == 'UV1T_to_UV' :
        # point is like [3, num], each column is [u, v, 1]^T
        # convert to [num, 2], each line is [u, v]
        point_ret = point.copy()[0:2, :].T
    elif CONVERT == 'TL_to_C' :
        # point is like [2, num] or [num, 2]
        #(0, 0) from top left to center
        #and point
        assert type(para['H']) == int, para
        assert type(para['W']) == int, para
        if para['direction'] == 'line' :
            # [num, 2]
            point_ret = point.copy()
            point_ret[:, 0] = point[:, 0] - para['W']/2
            point_ret[:, 1] = - point[:, 1] + para['H']/2
        elif para['direction'] == 'col' :
            # [2, num]
            point_ret = point.copy()
            point_ret[0, :] = point[0, :] - para['W']/2
            point_ret[1, :] = - point[1, :] + para['H']/2
        else :
            raise(NotImplementedError)
    elif CONVERT == 'C_to_TL' :
        #(0, 0) from center to top left
        assert type(para['H']) == int, para
        assert type(para['W']) == int, para
        if para['direction'] == 'line' :
            # [num, 2]
            point_ret = point.copy()
            point_ret[:, 0] = point[:, 0] + para['W']/2
            point_ret[:, 1] = - point[:, 1] + para['H']/2
        elif para['direction'] == 'col' :
            # [2, num]
            point_ret = point.copy()
            point_ret[0, :] = point[0, :] + para['W']/2
            point_ret[1, :] = - point[1, :] + para['H']/2
            pass
        else :
            raise(NotImplementedError)
    else :
        raise(NotImplementedError())
    return point_ret


def transform_rgb_image_for_saving(image:torch.Tensor) -> (np.ndarray) :
    '''
    input torch tensor[0-1], output numpy tensor[0-255]

    expect input like [512, 512, 3] not [3, 512, 512]

    content: 1. to cpu 2. to numpy 3. RGB to BGR
    '''
    image_cpu = (image*255.0).cpu()
    image_np = image_cpu.numpy().astype(np.uint8)
    image_mat_bgr = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
    return image_mat_bgr

def transform_gray_image_for_saving(image:torch.Tensor) -> (np.ndarray) :
    '''
    input torch tensor[0-1], output numpy tensor[0-255]

    expect input like [512, 512, 1] not [1, 512, 512]

    content: 1. to cpu 2. to numpy
    '''
    image_cpu = (image*255.0).cpu()
    image_np = image_cpu.numpy().astype(np.uint8)
    return image_np