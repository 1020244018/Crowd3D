import os
import sys
import numpy as np
import cv2
import pickle
import math

def compute_ankle(v1,v2, iters=2):
    xy=0.0
    xx=0.0
    yy=0.0
    for i in range(iters):
        xy+=v1[i]*v2[i]
        xx+=v1[i]**2
        yy+=v2[i]**2
    cos_value=xy/((xx*yy)**0.5 + 0.000000001)
    angle=math.acos(cos_value)
    return math.degrees(angle)

def compute_cos(v1,v2, iters=3):
    xy=0.0
    xx=0.0
    yy=0.0
    for i in range(iters):
        xy+=v1[i]*v2[i]
        xx+=v1[i]**2
        yy+=v2[i]**2
    cos_value=xy/((xx*yy)**0.5 + 0.000000001)

    return cos_value

def filter_single_kps(kps):
    '''
    kps: 17,3
    '''
    
    if kps is None:
        return True
    
    one_leg_tre=120
    two_legs_tre=260
    body_ankle_tre=160
    criterion1_flag=True
    criterion2_flag=True

    # criterion1 one leg > one_leg_tre, two legs > two_legs_tre
    if criterion1_flag:
        #print('start criterion1')
        left_leg_ankle=compute_ankle(kps[11]-kps[13], kps[15]-kps[13])
        right_leg_ankle=compute_ankle(kps[12]-kps[14], kps[16]-kps[14])
        if min(left_leg_ankle, right_leg_ankle) < one_leg_tre:
            return True
        if left_leg_ankle+right_leg_ankle < two_legs_tre:
            return True

    # criterion2 The trunk and average leg are collinear
    if criterion2_flag:
        #print('start criterion2')
        body_ankle=compute_ankle((kps[5]+kps[6])/2 - (kps[11]+kps[12])/2, (kps[15]+kps[16])/2 - (kps[11]+kps[12])/2)
        if body_ankle<body_ankle_tre:
            return True
    return False


def filter_single_kps_v2(kps):
    '''
    kps: 17,3
    '''
    
    if kps is None:
        return True
    
    one_leg_tre=120
    two_legs_tre=260
    body_ankle_tre=160
    criterion1_flag=True
    criterion2_flag=True

    # criterion1 one leg > one_leg_tre, two legs > two_legs_tre
    if criterion1_flag:
        #print('start criterion1')
        left_leg_ankle=compute_ankle(kps[11]-kps[13], kps[15]-kps[13])
        right_leg_ankle=compute_ankle(kps[12]-kps[14], kps[16]-kps[14])
        if min(left_leg_ankle, right_leg_ankle) < one_leg_tre:
            return True
        if left_leg_ankle+right_leg_ankle < two_legs_tre:
            return True

    # criterion2 The trunk and average leg are collinear
    if criterion2_flag:
        #print('start criterion2')
        body_ankle=compute_ankle((kps[5]+kps[6])/2 - (kps[11]+kps[12])/2, (kps[15]+kps[16])/2 - (kps[11]+kps[12])/2)
        if body_ankle<body_ankle_tre:
            return True
    
    # if cam hori
    body_direction=(kps[5]+kps[6])/2 - (kps[15]+kps[16])/2
    torso_direction=(kps[5]+kps[6])/2 - (kps[11]+kps[12])/2
    # bottom_body_direction=(kps[11]+kps[12])/2 - (kps[15]+kps[16])/2
    left_leg_direction=kps[11] - kps[15]
    right_leg_direction=kps[12] - kps[16]
    up_direction=(0, -1.)

    vertical_ankle1=compute_ankle(body_direction, up_direction)
    vertical_ankle2=compute_ankle(torso_direction, up_direction)
    left_leg_ankle=compute_ankle(left_leg_direction, up_direction)
    right_leg_ankle=compute_ankle(right_leg_direction, up_direction)
    if vertical_ankle1 >= 35 or vertical_ankle2 >=35:
        return True
    if left_leg_ankle >=35 or right_leg_ankle >=35:
        return True

    return False

def filter_kps(joint_2d):
    new_joint_2d=[]
    for i in range(len(joint_2d)):
        if not filter_single_kps(joint_2d[i]):
            new_joint_2d.append(joint_2d[i])
    return new_joint_2d

def filter_kps_v2(joint_2d):
    new_joint_2d=[]
    for i in range(len(joint_2d)):
        if not filter_single_kps_v2(joint_2d[i]):
            new_joint_2d.append(joint_2d[i])
    return new_joint_2d

# def filter_kps_mask(joint_2d, mask):
#     new_joint_2d=[]
#     for i in range(len(joint_2d)):
#         joint_mean=joint_2d[i].mean(0)
#         w, h =joint_mean[:2]
#         w, h=int(w), int(h)
#         height, width=mask.shape[0], mask.shape[1]
#         if w>=0 and w<width and h>=0 and h<height:
#             if mask[w, h]>0.00001:
#                 continue
#         new_joint_2d.append(joint_2d)
        
#     return new_joint_2d

def filter_kps_mask(joint_2d):
    new_joint_2d=[]
    for i in range(len(joint_2d)):
        joint_mean=joint_2d[i].mean(0)
        w, h =joint_mean[:2]
        w, h=int(w), int(h)
      
        if w< 3500:
            continue
        new_joint_2d.append(joint_2d[i])
        
    return new_joint_2d