
from typing import List
import numpy as np
import math
import cv2 as cv

def reverse_point(p1, p2, p3, p4):
    return p1, p4, p3, p2

def add_a_rec(list_v:List[np.ndarray], list_f:List[np.ndarray], list_c:List[np.ndarray], p1, p2, p3, p4, color):
    '''
    p1 +--+ p4
       |  |
    p2 +--+ p3
        ^y
        |
        |
    ----+---->x
        |
        |
    '''
    #p1, p2, p3, p4 = reverse_point(p1, p2, p3, p4)

    num_v = len(list_v)

    list_f.append(np.array([num_v, num_v + 1, num_v + 2], dtype=np.int32))
    list_f.append(np.array([num_v + 2, num_v + 3, num_v], dtype=np.int32))
    
    list_v.append(p1)
    list_v.append(p2)
    list_v.append(p3)
    list_v.append(p4)

    list_c.append(color)
    list_c.append(color)
    list_c.append(color)
    list_c.append(color)
    return

def add_an_arc(
        list_verts:List[np.ndarray], 
        list_faces:List[np.ndarray], 
        list_c:List[np.ndarray],
        color:np.ndarray,
        p1:np.ndarray, 
        p2:np.ndarray, 
        d:float, 
        normal:np.ndarray,
        divisions = 120
    ):
    '''
        list_verts: [,3]
        list_faces: [,3]
        d : 圆弧宽度
        normal: 法向量
        divisions: 割圆术割的次数
    '''
    # 偏移量
    base_pc = len(list_verts)   # base_pointcount 

    # 假设是半圆 
    delta = math.pi / (divisions - 1)
    p1 = np.array(p1,dtype=np.float32)
    p2 = np.array(p2,dtype=np.float32)
    center=(p1+p2)*0.5

    u = np.array(normal,dtype=np.float32)   # 单位法向量
    normal_std = normal / np.linalg.norm(normal, ord=2)
    
    # inner process
    v = p1 - center # 初始向量
    # 从 p1 开始转动180°
    
    inner_list = []
    out_list = []

    base = 0 
    for i in range(divisions):
        R, ____ = cv.Rodrigues(normal_std * delta * i)
        v_new = np.matmul(v, R.T)
        # 求得的是向量 
        point = v_new + center
        inner_list.append(point)
        list_verts.append(point)
        list_c.append(color)


    # 按比例计算向量
    ratio = 1 + d / math.sqrt(np.dot(v,v))
    v_out = ratio * v
    
    base = 0 
    for i in range(divisions):
        R, ____ = cv.Rodrigues(normal_std * delta * i)
        v_new = np.matmul(v_out, R.T)

        point = v_new + center
        out_list.append(point)
        list_verts.append(point)
        list_c.append(color)

     # inner stride = 2, out stride = 1
    for i in range(0,len(inner_list) -1):
        list_faces.append(
            np.array(
                [
                    base_pc + i,
                    base_pc + i + 1,
                    base_pc+ len(inner_list) + i
                ],
                dtype=np.int32
            )
        )

    # inner stride = 1, out stride = 2
    for i in range(0,len(out_list)-1):
        list_faces.append(
            np.array(
                [
                    base_pc + i +len(inner_list), 
                    base_pc + i + 1,
                    base_pc+len(inner_list)+i+1
                ],
                dtype=np.int32
            )
        )
