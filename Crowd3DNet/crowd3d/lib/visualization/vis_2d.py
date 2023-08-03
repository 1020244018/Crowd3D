import numpy as np
import cv2

def vis_joint_2d(img, joint,color_type='kps') :
    '''
    input:
        img
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

    for i, kps_ori in enumerate(joint):

        if kps_ori is None:
            continue
        
        kps = kps_ori[0:17,:]

        #添加文字
        """ xb = np.ones((2))
        for j in [0,1]:
            xb[j] = (kps[15][j] + kps[16][j]) / 2
        cv2.putText(
                img=img,
                text='%d' % i,
                org=(int(xb[0]),int(xb[1])),
                fontScale=1.5,
                thickness=2,
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                color=p_color[i%18]
        ) """ 


        part_line = {}

        # draw kps
        color = np.array(np.random.rand(3)) * 255
        per_kps = kps[:, :2]
        #kp_scores = kps[:, 2]
        circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.05) + 1
        for i, coord in enumerate(per_kps):
            x_coord, y_coord = int(coord[0]), int(coord[1])
            part_line[i] = (x_coord, y_coord)
            if color_type == 'kps':
                color = p_color[i]
            cv2.circle(img, (x_coord, y_coord), circle_size, color, -1)

        """ # draw position
        # 2d joint
        color = (255,0,0)
        cv2.circle(img, (int(xb[0]),int(xb[1])), circle_size+1, color, -1)
        # projection
        color = (0,255,0)
        cv2.circle(img, (int(kps_ori[17][0]),int(kps_ori[17][1])), circle_size, color, -1) """


        

        # draw limb
        limb_size = circle_size
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if color_type == 'kps':
                    color = p_color[i]
                if i < len(line_color):
                    cv2.line(img, start_xy, end_xy, color, limb_size)

    return img

def vis_point(img, points,color) :
    '''
    input:
        img
        points: numpy (num, *, 2) 
        color
    '''
    if color == 1:
        color_pro = (255,255,0)
    else:
        color_pro = (0, 255, 204)
    for i,point in enumerate(points):
        for j,coord in enumerate(point):
            x_coord, y_coord = int(coord[0]), int(coord[1])
            # print('投影得到的点：',x_coord, y_coord)
            cv2.circle(img, (x_coord, y_coord), 5, color_pro, -1)  #-1 实心 2 圈
    return img