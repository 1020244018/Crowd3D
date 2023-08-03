
from typing import Dict, List

import os

import cv2 as cv
import numpy as np

from lib.vis.vis_2d import vis_info, vis_points, vis_lines
from lib.solve.linear import solve_ground_from_xyz

from lib.util.transform import convert_point_vector
from lib.ground.util import uv_to_xyz_via_ground
from lib.geometry.plane import get_three_vanishing_point

LEN_COLOR = 9
COLOR = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (127, 0, 255),
    (127, 255, 0),
    (255, 127, 0),
    (255, 0, 127),
    (0, 127, 255),
    (0, 255, 127)
]

def vis_grid_from_homo(
        image_vis:np.ndarray,
        path_vis=str,
        matrix_homo=np.ndarray,
        flag_panda_lalel=False
    ):

    H, W, C = image_vis.shape
    #print(W, H)
    #exit()
    if flag_panda_lalel:
        #matrix_homo is the M of MI=G
        matrix_homo_T_inv = np.linalg.inv(matrix_homo.T)
    else:
        matrix_homo_T_inv = np.linalg.inv(matrix_homo.T)
        #matrix_homo_T_inv = np.linalg.inv(matrix_homo).T
    
    def _draw_line_from_image_coor(_point, _point_d, color=(180, 180, 180)):
        if _point_d[0] - _point[0] == 0 or _point_d[1] - _point[1] == 0:
            return False
        x1 = _point[0]
        y1 = _point[1]
        x2 = _point_d[0]
        y2 = _point_d[1]

        def _x(_y, _p1, _p2):
            return int((_y - _p1[1]) * (_p2[0] - _p1[0]) / (_p2[1] - _p1[1]) + _p1[0])
        def _y(_x, _p1, _p2):
            return int((_x - _p1[0]) * (_p2[1] - _p1[1]) / (_p2[0] - _p1[0]) + _p1[1])

        flag_border = []
        flag_border.append(0 <= _x(0, _point, _point_d) and _x(0, _point, _point_d) < W)
        flag_border.append(0 <= _y(0, _point, _point_d) and _y(0, _point, _point_d) < H)
        flag_border.append(0 <= _x(H-1, _point, _point_d) and _x(H-1, _point, _point_d) < W)
        flag_border.append(0 <= _y(W-1, _point, _point_d) and _y(W-1, _point, _point_d) < H)
        point_candidates = []
        point_candidates.append((_x(0, _point, _point_d), 0))
        point_candidates.append((0, _y(0, _point, _point_d)))
        point_candidates.append((_x(H-1, _point, _point_d), H-1))
        point_candidates.append((W-1, _y(W-1, _point, _point_d)))

        #print('---draw-begin---')
        #print(_point)
        #print(_point_d)
        #print(W, H)
        #print(point_candidates)
        #print(flag_border)

        p1 = None
        p2 = None
        indexes_proposal = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for indexes in indexes_proposal:
            if flag_border[indexes[0]] and flag_border[indexes[1]]:
                p1 = point_candidates[indexes[0]]
                p2 = point_candidates[indexes[1]]
                break
        if p1 is None or p2 is None:
            return False
    
        
        cv.line(
            image_vis, 
            pt1=p1,
            pt2=p2,
            color=color,
            thickness=2
        )
        return True

    def _draw_line_from_ground_coor(p1, d, color=(180, 180, 180)):
        coor_ground_homo = np.ones((2, 3))
        #point
        coor_ground_homo[0, 0] = p1[0]
        coor_ground_homo[0, 1] = p1[1]
        #direction
        coor_ground_homo[1, 0] = p1[0] + d[0]
        coor_ground_homo[1, 1] = p1[1] + d[1]
        coors_image_reprojected = np.matmul(coor_ground_homo, matrix_homo_T_inv)
        coors_image_reprojected = (coors_image_reprojected.T / coors_image_reprojected.T[2, :]).T
        point = coors_image_reprojected[0, 0:2]
        direction = coors_image_reprojected[1, 0:2]
        flag = _draw_line_from_image_coor(point, direction, color=color)
        return flag

    #x++
    for i in range(0, 200, 2):
        flag = _draw_line_from_ground_coor((i, 0), (0, 1), color=(200, 200, 0))
        if not flag:
            break
    for i in range(0, 200, 2):
        flag = _draw_line_from_ground_coor((-i, 0), (0, 1), color=(200, 200, 0))
        if not flag:
            break
    for i in range(0, 200, 2):
        flag = _draw_line_from_ground_coor((0, i), (1, 0), color=(200, 200, 0))
        if not flag:
            break
    for i in range(0, 200, 2):
        flag = _draw_line_from_ground_coor((0, -i), (1, 0), color=(200, 200, 0))
        if not flag:
            break

    v1, v2, v3 = get_three_vanishing_point(image_vis.shape, matrix_homo_T_inv)
    #v3[0] = 13870.53122042
    #v3[1] = 175443.9945
    num_line_v3 = 1080
    for i in range(num_line_v3):
        deg = 2 * np.pi / num_line_v3 * i
        direction = np.zeros((2))
        direction[0] = np.sin(deg)
        direction[1] = np.cos(deg)
        flag = _draw_line_from_image_coor(v3, v3 + direction, color=(0, 0, 200))
        #print(v3, direction, flag)
    cv.imwrite(path_vis, image_vis)
    #print(path_vis)
    #exit()
    #get_three_vanishing_point(image_vis.shape, matrix_homo_T_inv)
         

def vis_src_and_target_point(
    vis_data_pack:Dict,
    point_src_uv1t:np.ndarray,
    point_target_uv1t:np.ndarray,
    point_ref_uv1t:np.ndarray
    ) :

    image_vis = vis_data_pack['image'].copy()

    if 'prefix' in vis_data_pack.keys():
        name_image = vis_data_pack['prefix'] + '_' + 'src_and_target.jpg'
    else:
        name_image = 'src_and_target.jpg'
    path_vis_src_and_target = os.path.join(vis_data_pack['path'], name_image)

    point_src_uv_tl = convert_point_vector(
        convert_point_vector(point_src_uv1t, 'UV1T_to_UV'),
        'C_to_TL',
        {   'W': int(vis_data_pack['image'].shape[1] / vis_data_pack['scale']), 
            'H': int(vis_data_pack['image'].shape[0] / vis_data_pack['scale']),
            'direction': 'line'}
    )

    point_target_uv_tl = convert_point_vector(
        convert_point_vector(point_target_uv1t, 'UV1T_to_UV'),
        'C_to_TL',
        {   'W': int(vis_data_pack['image'].shape[1] / vis_data_pack['scale']), 
            'H': int(vis_data_pack['image'].shape[0] / vis_data_pack['scale']),
            'direction': 'line'}
    )

    point_ref_uv_tl = convert_point_vector(
        convert_point_vector(point_ref_uv1t, 'UV1T_to_UV'),
        'C_to_TL',
        {   'W': int(vis_data_pack['image'].shape[1] / vis_data_pack['scale']), 
            'H': int(vis_data_pack['image'].shape[0] / vis_data_pack['scale']),
            'direction': 'line'}
    )

    point_src_uv_tl = convert_point_vector(point_src_uv1t, 'UV1T_to_UV')
    point_target_uv_tl = convert_point_vector(point_target_uv1t, 'UV1T_to_UV')
    point_ref_uv_tl = convert_point_vector(point_ref_uv1t, 'UV1T_to_UV')

    vis_points(
        image_vis,
        point_src_uv_tl,
        scale=vis_data_pack['scale'],
        color=(0, 0, 255)
    )

    vis_points(
        image_vis,
        point_target_uv_tl,
        scale=vis_data_pack['scale'],
        color=(0, 255, 0)
    )

    vis_points(
        image_vis,
        point_ref_uv_tl,
        scale=vis_data_pack['scale'],
        color=(255, 0, 0)
    )

    vis_lines(
        image_vis,
        point_ref_uv_tl,
        point_target_uv_tl,
        scale=vis_data_pack['scale'],
        color=(255, 255, 0)
    )

    vis_lines(
        image_vis,
        point_ref_uv_tl,
        point_src_uv_tl,
        scale=vis_data_pack['scale'],
        color=(255, 0, 255)
    )


    cv.imwrite(path_vis_src_and_target, image_vis)

    return


def vis_point_with_info(
    vis_data_pack:Dict,
    point_uv1t:np.ndarray,
    info:List[str]
    ) :

    image_vis = vis_data_pack['image'].copy()

    path_vis_init_h_d = os.path.join(vis_data_pack['path'], 'init_h_d.jpg')

    point_uv = convert_point_vector(point_uv1t, 'UV1T_to_UV')
    '''
    point_uv_tl = convert_point_vector(
        point_uv,
        'C_to_TL',
        {   'W': int(vis_data_pack['image'].shape[1] / vis_data_pack['scale']), 
            'H': int(vis_data_pack['image'].shape[0] / vis_data_pack['scale']),
            'direction': 'line'}
    )
    '''

    vis_info(
        image_vis, 
        points=point_uv,
        info=info,
        scale=vis_data_pack['scale']
    )
    cv.imwrite(path_vis_init_h_d, image_vis)

    return


def vis_ground_grid(
    vis_data_pack:Dict,
    ground:np.ndarray, 
    cam_para:np.ndarray,
    grid_direction:str='parallel',
    point_uv1t:np.ndarray or None=None
    ) :
    '''
    input:
        * vis_data_pack = {
            'flag': True,
            'image': image_vis,
            'scale': ratio_scale_vis,
            'path': path_vis_folder
        } 
        * ground [4] np.ndarray
        * cam_para [3, 3] np.ndarray
        * grid_direction str 'parallel' or 'slash'
        * point_uv1t [3, n] np.ndarray, each col is [u, v, T]
    '''
    
    if grid_direction == 'slash' :
        assert point_uv1t is not None

        tmp = np.sum(point_uv1t, axis=0)

        top_right = point_uv1t[:, np.argmax(tmp):np.argmax(tmp) + 1]
        bot_left = point_uv1t[:, np.argmin(tmp):np.argmin(tmp) + 1]

        grid_tr = uv_to_xyz_via_ground(top_right, ground, cam_para)
        grid_bl = uv_to_xyz_via_ground(bot_left, ground, cam_para)

        N = ground[0:3].copy()

        rotate_x, ____ = cv.Rodrigues((np.pi / 4.0) * N / np.linalg.norm(N))
        rotate_y, ____ = cv.Rodrigues((np.pi / 4.0 * 7.0) * N / np.linalg.norm(N))

        grid_x_direction = np.matmul(
            rotate_x,
            grid_tr - grid_bl
        )
        grid_y_direction = np.matmul(
            rotate_y,
            grid_tr - grid_bl
        )

        point_zero = grid_bl
    elif grid_direction == 'parallel' :
        if point_uv1t is not None :
            Warning('Func:vis_ground_grid with not use point_uv1t when grid_direction == \'parallel\'')
        H_VIS = vis_data_pack['image'].shape[0] / vis_data_pack['scale']
        W_VIS = vis_data_pack['image'].shape[1] / vis_data_pack['scale']
        
        point_zero = uv_to_xyz_via_ground(
            np.array([[W_VIS * 0.5], [H_VIS * 0.9], [1]]), 
            ground, 
            cam_para
        )
        point_direct_x = uv_to_xyz_via_ground(
            np.array([[W_VIS], [H_VIS * 0.8925], [1]]), 
            ground, 
            cam_para
        )

        #print('p_zero', point_zero)
        #print(np.matmul(cam_para, point_zero) / np.matmul(cam_para, point_zero)[2])
        #exit()

        grid_x_direction = point_direct_x - point_zero
        N = ground[0:3].copy()
        rotate_y, ____ = cv.Rodrigues((np.pi / 2.0) * N / np.linalg.norm(N))
        grid_y_direction = np.matmul(
            rotate_y,
            grid_x_direction
        )
    else :
        raise(NotImplementedError(grid_direction))

    grid_x_direction = grid_x_direction / np.linalg.norm(grid_x_direction)
    grid_y_direction = grid_y_direction / np.linalg.norm(grid_y_direction)

    image_vis = vis_data_pack['image'].copy()

    def _draw_line(start_point, end_point, thickness=30) :
        start_uvz = np.matmul(cam_para, start_point)
        end_uvz = np.matmul(cam_para, end_point)
    
        start_uv = (start_uvz / start_uvz[2, 0]) * vis_data_pack['scale']
        end_uv = (end_uvz / end_uvz[2, 0]) * vis_data_pack['scale']

        #print(cam_para)
        #print(start_uv)
        #print(end_uv)
        #exit()

        '''
        start_uv = convert_point_vector(
            start_uv,
            'C_to_TL',
            {   'W': int(vis_data_pack['image'].shape[1]), 
                'H': int(vis_data_pack['image'].shape[0]),
                'direction': 'col'}
        )

        end_uv = convert_point_vector(
            end_uv,
            'C_to_TL',
            {   'W': int(vis_data_pack['image'].shape[1]), 
                'H': int(vis_data_pack['image'].shape[0]),
                'direction': 'col'}
        )
        '''

        cv.line(
            image_vis, 
            pt1=(int(start_uv[0, 0]), int(start_uv[1, 0])),
            pt2=(int(end_uv[0, 0]), int(end_uv[1, 0])),
            color=(213, 206, 91),
            thickness=thickness
        )
        return

    #num_line = 20
    num_line_x_down = 10
    num_line_x_up = 35
    #num_line_x = 25
    num_line_y = 12
    interval = 4
    for i in range(-num_line_y, num_line_y + 1) :
        start_point = point_zero + grid_x_direction * interval * i - grid_y_direction * interval * num_line_x_down
        end_point = point_zero + grid_x_direction * interval * i + grid_y_direction * interval * num_line_x_up

        #print('start_point', start_point)

        _draw_line(start_point=start_point, end_point=end_point, thickness=60)

   

    for i in range(-num_line_x_down, num_line_x_up + 1) :
        start_point = point_zero + grid_y_direction * interval * i - grid_x_direction * interval * num_line_y
        end_point = point_zero + grid_y_direction * interval * i + grid_x_direction * interval * num_line_y

        _draw_line(start_point=start_point, end_point=end_point)


    path_vis_init_ground = os.path.join(vis_data_pack['path'], '%s_ground.jpg' % vis_data_pack['prefix'])
    #print(path_vis_init_ground)
    cv.imwrite(path_vis_init_ground, image_vis)

    return


def vis_point(image, points) :
    #w = 3840
    #h = 2160
    h,w,c = image.shape
    len = points.shape[1]
    for i in range(len) :
        center = (int(points[0, i] + w/2), int(h/2 - points[1, i]))

        cv.circle(
            img=image, 
            center=center, 
            radius=10, 
            color=COLOR[i%LEN_COLOR],
            thickness=16
        )

        cv.putText(
                img=image,
                text='%d' % i,
                org=center,
                fontScale=1.5,
                thickness=2,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                color=COLOR[i%LEN_COLOR]
        )
    return


def vis_point_distance(image, points, points3d) :
    #w = 3840
    #h = 2160
    h,w,c = image.shape
    len = points.shape[1]
    count = -1
    for i in range(3) :
        for j in range(i + 1, len) :
            count = count + 1

            distance = np.sqrt(
                np.sum(
                    (points3d[:, i] - points3d[:, j]) ** 2
                )
            )

            pt1=int(points[0, i] + w/2), int(h/2 - points[1, i])
            pt2=int(points[0, j] + w/2), int(h/2 - points[1, j])
            pt_center = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
            cv.line(
                img=image,
                pt1=pt1,
                pt2=pt2,
                color=COLOR[count%LEN_COLOR],
                thickness=4
            )
            cv.putText(
                img=image,
                text='%.2f' % distance,
                org=pt_center,
                fontScale=1,
                thickness=2,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                color=COLOR[count%LEN_COLOR]
            )
    return


def vis_ground_as_prism(
    point:np.ndarray, 
    ground:np.ndarray, 
    path:str
    ) :
    '''
    input :
        * point [3, n] ndarray
        * ground [4] ndarray
        * path
    '''
    assert len(point.shape) == 2
    assert point.shape[0] == 3

    num = point.shape[1]

    normal = ground[0:3].copy()
    normal = normal / np.sqrt(np.sum(normal ** 2))
    #print(normal)
    #np.save('./output/seq02_001/ground_normal.npy',normal)


    with open(path, 'w') as f :
        for i in range(num) :
            v1 = point[:, i]
            v_tmp = point[:, i] + normal * 1.7

            v2 = v_tmp.copy()
            v2[2] = v2[2] - 0.5

            v3 = v_tmp.copy()
            v3[0] = v3[0] + 0.5

            v4 = v_tmp.copy()
            v4[2] = v4[2] + 0.5

            v5 = v_tmp.copy()
            v5[0] = v5[0] - 0.5

            f.write('v %f %f %f\n' % (v1[0], v1[1], v1[2]))
            f.write('v %f %f %f\n' % (v2[0], v2[1], v2[2]))
            f.write('v %f %f %f\n' % (v3[0], v3[1], v3[2]))
            f.write('v %f %f %f\n' % (v4[0], v4[1], v4[2]))
            f.write('v %f %f %f\n' % (v5[0], v5[1], v5[2]))

            '''
            f.write('v %f %f %f\n' % (v1[0], v1[1], v1[2]))
            f.write('v %f %f %f\n' % (v2[0], v2[1], v2[2]))
            f.write('v %f %f %f\n' % (v3[0], v3[1], v3[2]))
            f.write('v %f %f %f\n' % (v4[0], v4[1], v4[2]))
            f.write('v %f %f %f\n' % (v5[0], v5[1], v5[2])) 
            '''
            

        for i in range(num) :
            l = 5 * i
            f.write('f %d %d %d\n' % (l+1, l+2, l+3))
            f.write('f %d %d %d\n' % (l+1, l+3, l+4))
            f.write('f %d %d %d\n' % (l+1, l+4, l+5))
            f.write('f %d %d %d\n' % (l+1, l+5, l+2))
            f.write('f %d %d %d\n' % (l+2, l+3, l+4))
            f.write('f %d %d %d\n' % (l+4, l+5, l+2))  
           
    return

def vis_ground_plane(ground_point, save_scene_path):
    obj_name = (os.path.join(save_scene_path, '1_ground_z.obj'))
    print('obj_name',obj_name)

    with open(obj_name, 'w') as f:
        
        for i in range(8) :
            f.write('v %f %f %f\n' % (ground_point[i][0], ground_point[i][1], -ground_point[i][2]))
        
        f.write('f %d %d %d\n' % (1, 3, 2))
        f.write('f %d %d %d\n' % (1, 4, 3))
        f.write('f %d %d %d\n' % (5, 6, 7))
        f.write('f %d %d %d\n' % (5, 7, 8))
        f.write('f %d %d %d\n' % (1, 4, 8))
        f.write('f %d %d %d\n' % (1, 8, 5))
        f.write('f %d %d %d\n' % (2, 7, 3))
        f.write('f %d %d %d\n' % (2, 6, 7))
        f.write('f %d %d %d\n' % (1, 5, 6))
        f.write('f %d %d %d\n' % (1, 6, 2))
        f.write('f %d %d %d\n' % (4, 3, 7))
        f.write('f %d %d %d\n' % (4, 7, 8))