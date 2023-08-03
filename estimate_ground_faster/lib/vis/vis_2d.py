
from typing import List, Tuple

import numpy as np
import cv2 as cv

def vis_lines(
    image_vis:np.ndarray, 
    points_start:np.ndarray,
    points_end:np.ndarray,
    scale:float=1.0,
    color:Tuple[int, int, int]=(255, 0, 0)
    ) :
    '''
    input:
        image [w, h, c] np.ndarray
        points_start [num, 2] np.ndarray
        points_end [num, 2] np.ndarray
        info List of length num
    '''
    assert len(points_start.shape) == 2, points_start.shape
    assert points_start.shape[1] == 2, points_start.shape
    assert points_start.shape == points_end.shape, points_end.shape

    thickness = 4

    for i in range(points_start.shape[0]) :
        position_start = (
            int(points_start[i, 0] * scale),
            int(points_start[i, 1] * scale)
        )

        position_end = (
            int(points_end[i, 0] * scale),
            int(points_end[i, 1] * scale)
        )

        cv.line(
            image_vis,
            pt1=position_start,
            pt2=position_end,
            color=color,
            thickness=thickness
        )
    return None


def vis_points(
    image_vis:np.ndarray, 
    points:np.ndarray,
    scale:float=1.0,
    color:Tuple[int, int, int]=(255, 0, 0)
    ) :
    '''
    input:
        image [w, h, c] np.ndarray
        points [num, 2] np.ndarray
        info List of length num
    '''
    assert len(points.shape) == 2
    assert points.shape[1] == 2

    radius = 4

    for i in range(points.shape[0]) :
        position = (
            int(points[i, 0] * scale),
            int(points[i, 1] * scale)
        )
        cv.circle(
            image_vis, 
            center=position, 
            radius=radius, 
            color=color, 
            thickness=-1
        )
    return None


def vis_info(
    image_vis:np.ndarray, 
    points:np.ndarray,
    info:List[str],
    scale:float=1.0,
    color:Tuple[int, int, int]=(255, 0, 0)
    ) -> (np.ndarray) :
    '''
    input:
        image [w, h, c] np.ndarray
        points [num, 2] np.ndarray
        info List of length num
    '''
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert points.shape[0] == len(info)

    for i in range(points.shape[0]) :
        position = (
            int(points[i, 0] * scale),
            int(points[i, 1] * scale)
        )
        cv.putText(
            img=image_vis,
            text=info[i],
            org=position,
            fontScale=1,
            thickness=2,
            fontFace=cv.FONT_HERSHEY_COMPLEX,
            color=color
        )
    return None


def vis_joint_2d(
    img_vis:np.ndarray, 
    joint:np.ndarray, 
    color_type:str='kps'
    ) :
    '''
    input:
        img_vis
        joint_2d: numpy (person_num, coco_17, 3)
    return:
        img with 2d joint
    '''
    format = 'coco'
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    for i, kps in enumerate(joint):

        if kps is None:
            continue

        part_line = {}

        # draw kps
        color = np.array(np.random.rand(3)) * 255
        per_kps = kps[:, :2]
        kp_scores = kps[:, 2]
        circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.05) + 1
        for i, coord in enumerate(per_kps):
            x_coord, y_coord = int(coord[0]), int(coord[1])
            part_line[i] = (x_coord, y_coord)
            if color_type == 'kps':
                color = p_color[i]
            cv.circle(img_vis, (x_coord, y_coord), circle_size, color, -1)

        # draw limb
        limb_size = circle_size
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if color_type == 'kps':
                    color = p_color[i]
                if i < len(line_color):
                    cv.line(img_vis, start_xy, end_xy, color, limb_size)

    return img_vis
    