from modulefinder import Module
import numpy as np
import torch
from lib.ground.util import uv_to_xyz_via_ground
from scipy.optimize import fsolve
# from lib.ground.homography import get_cam_param_and_RT_via_torch_opt

def get_line_coffe(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    c1 = y1 - y2
    c2 = -(x1 - x2)
    c3 = (y1 - y2)*x1 - (x1 - x2)*y1
    return c1, c2, c3

def get_intersection(p1, p2, p3, p4, MODULE:Module) -> (np.ndarray):
    assert MODULE == torch or np
    c01, c02, c03 = get_line_coffe(p1, p2)
    c11, c12, c13 = get_line_coffe(p3, p4)
    A = MODULE.zeros((2, 2))
    B = MODULE.zeros((2, 1))
    A[0, 0] = c01
    A[0, 1] = c02
    B[0, 0] = c03
    A[1, 0] = c11
    A[1, 1] = c12
    B[1, 0] = c13
    if MODULE == torch:
        x = MODULE.matmul(MODULE.inverse(A), B).T[0]
    elif MODULE == np:
        x = MODULE.matmul(MODULE.linalg.inv(A), B).T[0]
    else:
        raise(NotImplementedError(MODULE))
    return x

def get_normal_from_vec(vec:np.ndarray or torch.Tensor, flag_clockwise:bool=True):
    if type(vec) == np.ndarray:
        MODULE = np
    elif type(vec) == torch.Tensor:
        MODULE = torch
    else:
        raise(NotImplementedError(type(vec)))

    if MODULE == torch:
        ret = vec.clone()
    elif MODULE == np:
        ret = vec.copy()
    if flag_clockwise:
        ret[0] = vec[1]
        ret[1] = - vec[0]
    else:
        ret[0] = - vec[1]
        ret[1] = vec[0]
    return ret

def get_third_point(v1:np.ndarray, v2:np.ndarray, orthocenter:np.ndarray):
    if type(v1) == np.ndarray:
        MODULE = np
    elif type(v1) == torch.Tensor:
        MODULE = torch
    else:
        raise(NotImplementedError(type(v1)))

    assert v1.shape == (2,), v1.shape
    assert v2.shape == (2,), orthocenter.shape
    assert orthocenter.shape == (2,), orthocenter.shape

    if MODULE == torch:
        orthocenter_tmp = orthocenter.clone()
    elif MODULE == np:
        orthocenter_tmp = orthocenter.copy()
    else:
        raise(NotImplementedError(MODULE))
        
    orthocenter_tmp[0] = orthocenter[1]
    orthocenter_tmp[1] = orthocenter[0]
    OA = v1 - orthocenter_tmp
    OB = v2 - orthocenter_tmp
    #print(v1, v2, orthocenter_tmp)

    AC = get_normal_from_vec(OB)
    BC = get_normal_from_vec(OA)

    A2 = v1 + AC
    B2 = v2 + BC

    C = get_intersection(v1, A2, v2, B2, MODULE)

    #-f^2
    minus_f2 = MODULE.dot(OA, OB)
    if minus_f2 > 0:
        f = MODULE.sqrt(minus_f2)
        #print('minus_f2 > 0, and f =', MODULE.sqrt(minus_f2))
    else:
        f = MODULE.sqrt(-minus_f2)
        #print('minus_f2 < 0, and f =', MODULE.sqrt(-minus_f2))
    return C, f

def get_f(matrix_homo_T_inv, orthocenter):
    if type(matrix_homo_T_inv) == np.ndarray:
        assert type(orthocenter) == np.ndarray, type(orthocenter)
        MODULE = np
    elif type(matrix_homo_T_inv) == torch.Tensor:
        assert type(orthocenter) == torch.Tensor, type(orthocenter)
        MODULE = torch
    else:
        raise(NotImplementedError(type(matrix_homo_T_inv)))
    coor_ground_homo = MODULE.ones((8, 3))
    #point
    #line1
    coor_ground_homo[0, 0] = 0
    coor_ground_homo[0, 1] = 0
    coor_ground_homo[1, 0] = 0
    coor_ground_homo[1, 1] = 10
    #line2
    coor_ground_homo[2, 0] = 10
    coor_ground_homo[2, 1] = 0
    coor_ground_homo[3, 0] = 10
    coor_ground_homo[3, 1] = 10
    #line3
    coor_ground_homo[4, 0] = 0
    coor_ground_homo[4, 1] = 0
    coor_ground_homo[5, 0] = 10
    coor_ground_homo[5, 1] = 0
    #line4
    coor_ground_homo[6, 0] = 0
    coor_ground_homo[6, 1] = 10
    coor_ground_homo[7, 0] = 10
    coor_ground_homo[7, 1] = 10
    coors_image_reprojected = MODULE.matmul(coor_ground_homo, matrix_homo_T_inv)
    coors_image_reprojected = (coors_image_reprojected.T / coors_image_reprojected.T[2, :]).T
    
    v1 = get_intersection(
        (coors_image_reprojected[0, 0], coors_image_reprojected[0, 1]),
        (coors_image_reprojected[1, 0], coors_image_reprojected[1, 1]),
        (coors_image_reprojected[2, 0], coors_image_reprojected[2, 1]),
        (coors_image_reprojected[3, 0], coors_image_reprojected[3, 1]),
        MODULE
    )

    v2 = get_intersection(
        (coors_image_reprojected[4, 0], coors_image_reprojected[4, 1]),
        (coors_image_reprojected[5, 0], coors_image_reprojected[5, 1]),
        (coors_image_reprojected[6, 0], coors_image_reprojected[6, 1]),
        (coors_image_reprojected[7, 0], coors_image_reprojected[7, 1]),
        MODULE
    )

    v3, f = get_third_point(v1, v2, orthocenter)
    return f

def get_three_vanishing_point(image_shape, matrix_homo_T_inv):

    coor_ground_homo = np.ones((8, 3))
    #point
    #line1
    coor_ground_homo[0, 0] = 0
    coor_ground_homo[0, 1] = 0
    coor_ground_homo[1, 0] = 0
    coor_ground_homo[1, 1] = 10
    #line2
    coor_ground_homo[2, 0] = 10
    coor_ground_homo[2, 1] = 0
    coor_ground_homo[3, 0] = 10
    coor_ground_homo[3, 1] = 10
    #line3
    coor_ground_homo[4, 0] = 0
    coor_ground_homo[4, 1] = 0
    coor_ground_homo[5, 0] = 10
    coor_ground_homo[5, 1] = 0
    #line4
    coor_ground_homo[6, 0] = 0
    coor_ground_homo[6, 1] = 10
    coor_ground_homo[7, 0] = 10
    coor_ground_homo[7, 1] = 10
    coors_image_reprojected = np.matmul(coor_ground_homo, matrix_homo_T_inv)
    coors_image_reprojected = (coors_image_reprojected.T / coors_image_reprojected.T[2, :]).T
    
    v1 = get_intersection(
        (coors_image_reprojected[0, 0], coors_image_reprojected[0, 1]),
        (coors_image_reprojected[1, 0], coors_image_reprojected[1, 1]),
        (coors_image_reprojected[2, 0], coors_image_reprojected[2, 1]),
        (coors_image_reprojected[3, 0], coors_image_reprojected[3, 1]),
        MODULE=np
    )

    v2 = get_intersection(
        (coors_image_reprojected[4, 0], coors_image_reprojected[4, 1]),
        (coors_image_reprojected[5, 0], coors_image_reprojected[5, 1]),
        (coors_image_reprojected[6, 0], coors_image_reprojected[6, 1]),
        (coors_image_reprojected[7, 0], coors_image_reprojected[7, 1]),
        MODULE=np
    )

    orthocenter = np.array(image_shape[0:2]) / 2

    v3, f = get_third_point(v1, v2, orthocenter)
    #print('v3 and f')
    #print(v3, f)
    v3_homo = np.ones((3))
    v3_homo[0:2] = v3

    return v1, v2, v3
    
    #TODO:Vis V3
    #print(image_shape, v3_homo, normal_ground)
    #print(np.linalg.inv(matrix_cam))
    #print(matrix_homo_T_inv)


    matrix_homo_ground_to_image = matrix_homo_T_inv.T
    para_init = np.zeros((3))
    para_init[0] = 27000
    #para_init[1] = image_shape[1] / 2 * 1.06
    #para_init[2] = image_shape[0] / 2 * 1.05
    para_init[1] = image_shape[1] / 2
    para_init[2] = image_shape[0] / 2
    #get_cam_param_and_ground_via_torch_opt(matrix_homo_ground_to_image, para_init)
    def func(PARA):
        _matrix_cam = np.zeros((3, 3))#np.diag([F, F, 1],)
        _matrix_cam[0, 0] = PARA
        _matrix_cam[1, 1] = PARA
        _matrix_cam[0, 2] = image_shape[1] / 2
        _matrix_cam[1, 2] = image_shape[0] / 2
        _matrix_cam[2, 2] = 1
        _matrix_cam_inv = np.linalg.inv(_matrix_cam)
        _matrix_RT = np.matmul(_matrix_cam_inv, matrix_homo_ground_to_image)

        tmp_1 = np.sqrt(np.sum(_matrix_RT[:, 0] ** 2))
        tmp_2 = np.sqrt(np.sum(_matrix_RT[:, 1] ** 2))

        term_1 = np.abs(tmp_1 - 1)
        term_2 = np.abs(tmp_2 - 1)
        term_3 = np.abs(np.dot(_matrix_RT[:, 0], _matrix_RT[:, 1]))

        loss = np.ones((3))
        loss[0] = term_1
        loss[1] = term_2
        loss[2] = term_3

        return term_3

    
    root = fsolve(func, para_init[0])
    print('para_init', para_init)
    print('root', root)

    matrix_cam = np.zeros((3, 3))#np.diag([F, F, 1],)
    matrix_cam[0, 0] = root
    matrix_cam[1, 1] = root
    matrix_cam[0, 2] = image_shape[1] / 2
    matrix_cam[1, 2] = image_shape[0] / 2
    matrix_cam[2, 2] = 1
    matrix_RT = np.matmul(np.linalg.inv(matrix_cam), matrix_homo_ground_to_image)

    matrix_RT = 2 * matrix_RT / (np.sqrt(np.sum(matrix_RT[:, 0] ** 2)) + np.sqrt(np.sum(matrix_RT[:, 1] ** 2)))
    
    print(matrix_RT)
    print(np.sqrt(np.sum(matrix_RT[:, 0] ** 2)))
    print(np.sqrt(np.sum(matrix_RT[:, 1] ** 2)))
    print(np.dot(matrix_RT[:, 0], matrix_RT[:, 1]))
    
    np.save('./RT.npy', matrix_RT)
    exit()

    '''
    sample_point_cam_init = np.ones((3, 4))
    #point1
    sample_point_cam_init[0, 0] = 100
    sample_point_cam_init[1, 0] = 100
    #point2
    sample_point_cam_init[0, 1] = 100
    sample_point_cam_init[1, 1] = 150
    #point3
    sample_point_cam_init[0, 2] = 150
    sample_point_cam_init[1, 2] = 150
    #point4
    sample_point_cam_init[0, 3] = 100
    sample_point_cam_init[1, 3] = 150

    ground = np.ones((4))
    ground[0:3] = normal_ground
    ground[3] = - 10

    matrix_homo_ground_to_image = matrix_homo_T_inv.T
    sample_point_image_coor = np.matmul(matrix_homo_ground_to_image, sample_point_cam_init)
    sample_point_image_coor = sample_point_image_coor / sample_point_image_coor[2, :]
    print('sample_point_cam_init')
    print(sample_point_cam_init)
    print('sample_point_image_coor')
    print(sample_point_image_coor)
    print('ground and cam')
    print(ground)
    print(matrix_cam)
    sample_point_cam_coor = uv_to_xyz_via_ground(sample_point_image_coor, ground, matrix_cam)
    print('sample_point_cam_coor')
    print(sample_point_cam_coor)
    p1 = sample_point_cam_coor[:, 0]
    p2 = sample_point_cam_coor[:, 1]
    p3 = sample_point_cam_coor[:, 2]
    p4 = sample_point_cam_coor[:, 3]
    print(np.linalg.norm(p1 - p2, ord=2))
    print(np.linalg.norm(p2 - p3, ord=2))
    print(np.linalg.norm(p3 - p4, ord=2))
    print(np.linalg.norm(p4 - p1, ord=2))
    exit()
    '''


    
    matrix_homo_ground_to_image = matrix_homo_T_inv.T
    
    sample_point_cam_init = np.ones((3, 4))
    #point1
    sample_point_cam_init[0, 0] = 10
    sample_point_cam_init[1, 0] = 10
    #point2
    sample_point_cam_init[0, 1] = 10
    sample_point_cam_init[1, 1] = 30
    #point3
    sample_point_cam_init[0, 2] = 30
    sample_point_cam_init[1, 2] = 30
    #point4
    sample_point_cam_init[0, 3] = 30
    sample_point_cam_init[1, 3] = 10
    sample_point_cam_coor = np.matmul(matrix_RT, sample_point_cam_init)
    sample_point_image_coor = np.matmul(matrix_cam, sample_point_cam_coor)
    sample_point_image_coor = sample_point_image_coor / sample_point_image_coor[2, :]
    

    p1 = sample_point_cam_coor[:, 0]
    p2 = sample_point_cam_coor[:, 1]
    p3 = sample_point_cam_coor[:, 2]
    p4 = sample_point_cam_coor[:, 3]
    print(np.linalg.norm(p1 - p2, ord=2))
    print(np.linalg.norm(p2 - p3, ord=2))
    print(np.linalg.norm(p3 - p4, ord=2))
    print(np.linalg.norm(p4 - p1, ord=2))
    print(np.linalg.norm(p2 - p4, ord=2))
    print(np.linalg.norm(p1 - p3, ord=2))
    print(sample_point_cam_coor)
    print(sample_point_image_coor)

    sample_point_image_coor = np.matmul(matrix_homo_ground_to_image, sample_point_cam_init)
    sample_point_image_coor = sample_point_image_coor / sample_point_image_coor[2, :]
    #print(sample_point_image_coor)
    exit()
    
    #print(v1, v2, v3)
    #print(matrix_cam)
    
    
    

