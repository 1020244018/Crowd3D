

from typing import List, Dict, Tuple

import numpy as np
import cv2 as cv
import os


import torch

def get_cam_in_matrix(f:float, W, H):
    cam_in = np.array(
        [[f, 0, W/2],
        [0, f, H/2],
        [0, 0, 1]], dtype=np.float32
    )
    return cam_in

def if_on_the_ground(xyz:np.ndarray, ground:np.ndarray) :
    '''
    input :
        * xyz [n, 3]
        * ground [4] A, B,C,D
    '''

    assert len(xyz.shape) == 2
    assert xyz.shape[1] == 3
    
    return xyz[:, 0] * ground[0] + xyz[:, 1] * ground[1] + xyz[:, 2] * ground[2] + ground[3]


def uv_to_xyz_via_ground(uv:np.ndarray, ground:np.ndarray, cam_in:np.ndarray) :
    '''
    input:
        * uv [3, n] each col is [u, v, 1] or [2, n] each col is [u, v]
        * ground [4] A, B, C, D
        * cam in [3, 3]
    output 
        * xyz [3, n] each col is [x, y, z]
    '''
    assert len(uv.shape) == 2
    assert uv.shape[0] == 3 or uv.shape[0] == 2
    #assert uv[2, :] == np.ones_like(uv[2, :])

    fx = cam_in[0, 0]
    fy = cam_in[1, 1]
    cx = cam_in[0, 2]
    cy = cam_in[1, 2]
    
    num = uv.shape[1]
    new_z = np.zeros((num))

    for i in range(num) :
        new_z[i] = - ground[3] / (ground[0] * (1/fx) * (uv[0, i] - cx) + ground[1] * (1/fy) * (uv[1, i] - cy) + ground[2])
    
    xyz = np.matmul(np.linalg.inv(cam_in), uv * new_z)

    return xyz

def uv_to_xyz_via_ground_torch(uv:torch.Tensor, ground:torch.Tensor, cam_in:torch.Tensor) :
    '''
    input:
    Note all in the same device
        * uv [3, n] each col is [u, v, 1]
        * ground [4] A, B, C, D
        * cam in [3, 3]
    output 
        * xyz [3, n] each col is [x, y, z]
    '''
    assert len(uv.shape) == 2
    assert uv.shape[0] == 3

    fx = cam_in[0, 0]
    fy = cam_in[1, 1]
    cx = cam_in[0, 2]
    cy = cam_in[1, 2]

    num = uv.shape[1]
    new_z = torch.zeros((num), device=uv.device)

    for i in range(num) :
        new_z[i] = - ground[3] / (ground[0] * (1/fx) * (uv[0, i] - cx) + ground[1] * (1/fy) * (uv[1, i] - cy) + ground[2])
    
    xyz = torch.matmul(torch.inverse(cam_in), (uv * new_z))
    return xyz


class Labeled_Points_For_Checking():
    def __init__(self, list_points:List[List[float]], list_labels:List[Tuple[int, int, float]]):
        self.list_points = list_points
        self.list_labels = list_labels
        
    def get_point3d(self, ground:np.ndarray, cam_para:np.ndarray):
        list_point3d = []
        for point in self.list_points:
            assert type(point) == list
            assert len(point) == 2
            point_3d = uv_to_xyz_via_ground(
                np.array([[point[0]], [point[1]], [1]]), 
                ground, 
                cam_para
            )
            list_point3d.append(point_3d)
        return list_point3d

    def get_pred(self, ground:np.ndarray, cam_para:np.ndarray):
        list_point3d = self.get_point3d(ground=ground, cam_para=cam_para)
        list_gt = []
        list_pred = []
        for label in self.list_labels:
            indexA = label[0]
            indexB = label[1]
            len_gt = label[2]

            assert type(indexA) == int
            assert type(indexB) == int
            assert type(len_gt) == float

            list_gt.append(len_gt)
            pA = list_point3d[indexA]
            pB = list_point3d[indexB]

            list_pred.append(np.linalg.norm(pA - pB, ord=2))

        return list_gt, list_pred

        





lpfc_PANDA_shenzhenbei = Labeled_Points_For_Checking(
    list_points=[
        [18021, 9775],
        [20448, 8402],
        [18045.75, 5811.00],
        [18951.01, 5649.21]
    ],
    list_labels=[
        (0, 1, 12.0),
        (2, 3, 12.0)
    ]
)

