
import json
import os
from typing import Dict, List

import cv2 as cv
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm

from lib.ground.util import uv_to_xyz_via_ground
from lib.io.io_3d import load_mesh, write_obj, write_obj_color_int
from lib.util.transform import uv_to_uv1T
from lib.vis.draw_3d import add_a_rec, add_an_arc


def ground_to_obj(camera:np.ndarray, ground:np.ndarray):

    W = 19200
    H = 6480

    points = np.array([
        [0, 0],
        [0, H],
        [W, H],
        [W, 0]
    ]).T

    points_3d = uv_to_xyz_via_ground(points, ground=ground, cam_in=camera)

    list_v = []
    list_f = []
    list_c = []

    add_a_rec(
        list_v, list_f, list_c, 
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2], 
        points_3d[:, 3], 
        color=[250, 250, 250]
    )

    verts = np.array(list_v, dtype=np.float32)
    faces = np.array(list_f, dtype=np.int32)
    colors = np.array(list_c, dtype=np.uint8)
    write_obj_color_int(verts, colors, faces, './tmp_ground.obj')


def landmark_to_obj_playground1_00(landmark_image:Dict[str, np.ndarray], camera:np.ndarray, ground:np.ndarray):
    '''
    landmark_image: Dict
        dict_keys([
        'vec_y', 'vec_x', 
        'field_edge_x', 'field_edge_y', 
        'track_inner_edge_closer', 
        'track_inner_edge_oppo', 
        'track_inner_curve_start'
    ])
    '''
    #dict_keys(['trackline_edge', 'vec_x', 'vec_y', 'stand_edge_1', 'field_edge_y', 'field_edge_x'])
    list_v = []
    list_f = []
    list_c = []
    
    #get_vec
    def get_direction_from_line(line:np.ndarray):
        vec = uv_to_xyz_via_ground(uv_to_uv1T(line), ground=ground, cam_in=camera)
        vec = vec[:, 1] - vec[:, 0]
        vec = vec / np.linalg.norm(vec, ord=2)
        return vec
    vec_x = get_direction_from_line(landmark_image['vec_x'])
    vec_y = get_direction_from_line(landmark_image['vec_y'])
    vec_z = ground[0:3] / np.linalg.norm(ground[0:3])
    if vec_z[1] > 0:
        vec_z *= -1.0

    #process field label first
    field_edge_x = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['field_edge_x']), ground=ground, cam_in=camera)
    field_edge_y = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['field_edge_y']), ground=ground, cam_in=camera)

    point_start_std = field_edge_y[:, 0]
    point_oppo = field_edge_x[:, 1]

    
    vec_st_op = point_oppo - point_start_std
    width_field = np.dot(vec_st_op, vec_x)
    vec_st_op_std = vec_x * width_field
    point_oppo_std = point_start_std + vec_st_op_std

    HEIGHT_RIZE = 0.02

    #rise a little
    point_start_std += vec_z * HEIGHT_RIZE
    point_oppo_std += vec_z * HEIGHT_RIZE

    num_rec = field_edge_y.shape[1] - 1
    '''
                        +-------------+-------------+ 
                        |             |             |
                        |             |             |
                        |             |             |
                        |             |             | 
                        |             |             |
                        |             |             |
                        |             |             |
       field_line_start +-------------+-------------+ field_line_end
                          ************
    the half soccer court has 13 rec
    and only the 12 rec of the left half court are labeled,
    so this drawing code complete 26 rec from these 12 labled (see above '*')
    '''
    steps_labeled = np.zeros([num_rec])
    for i in range(num_rec):
        vec_step = field_edge_y[:, i + 1] - field_edge_y[:, i]
        steps_labeled[i] = np.dot(vec_step, vec_y)
    steps_26 = np.ones([26]) * np.average(steps_labeled)
    steps_26[1:1+num_rec] = steps_labeled

    field_line_start = point_start_std - vec_y * steps_26[0]
    field_line_end = point_start_std.copy()
    for i in range(1, 26): 
        field_line_end += vec_y * steps_26[i]

    #get_track
    def get_track():
        '''
         point4 +----+ point3
                |    |
                |    |
                |    |
          line2 |    | line1
                |    |
                |    |
                |    |
         point2 +----+ point1

                        trackline_start_oppo       trackline_end_oppo
                              +-----------------------+
                            +-------------+-------------+ 
                            |             |             |
                            |             |             |               ^ 
                            |             |             |               | vec_linecross
                            |             |             |               
                            |             |             |
                            |             |             |
                            |             |             |
        field_line_start    +-------------+-------------+ field_line_end  
                              +-----------------------+                   |length_field_track_height_bias
                              trackline_start     trackline_end
                              (curve begin)
                                                        -
                                                        length_field_track_width_bias
        '''
        trackline_start = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['track_inner_curve_start']), ground=ground, cam_in=camera)
        trackline_start = trackline_start[:, 0] #[1, 3] to [3]

        trackline_start -= HEIGHT_RIZE
        
        length_field_track_width_bias = np.dot(field_line_start - trackline_start, vec_y)
        trackline_end = trackline_start + (field_line_end - field_line_start) + length_field_track_width_bias * 2 * vec_y

        vec_linecross = vec_x * 1.12 * 10
        if vec_linecross[2] > 0:
            vec_linecross *= -1.0
        width_half_white_line = 0.04
        num_track = 10

        COLOR_TRACK = [212, 64, 64]
        COLOR_TRACK_INTERVAL = [250, 250, 250]

        #track this side
        for i in range(num_track):
            add_a_rec(
                list_v, list_f, list_c, 
                trackline_end + vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_start + vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_start + vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_end + vec_linecross*i/10 - vec_x * width_half_white_line,
                color=COLOR_TRACK_INTERVAL
            )
            
            add_a_rec(
                list_v, list_f, list_c, 
                trackline_end + vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_start + vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_start + vec_linecross*(i+1)/10 + vec_x * width_half_white_line, 
                trackline_end + vec_linecross*(i+1)/10 + vec_x * width_half_white_line,
                color=COLOR_TRACK
            )
        for i in [num_track]:
            add_a_rec(
                list_v, list_f, list_c, 
                trackline_end + vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_start + vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_start + vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_end + vec_linecross*i/10 - vec_x * width_half_white_line,
                color=COLOR_TRACK_INTERVAL
            )

        length_field_track_height_bias = np.dot(field_line_start - trackline_start, vec_x)
        width_field = np.dot(vec_st_op, vec_x)

        trackline_start_oppo = trackline_start + vec_x * (width_field + length_field_track_height_bias * 2)
        trackline_end_oppo = trackline_start_oppo + (trackline_end - trackline_start)

        
        for i in range(num_track):
            add_a_rec(
                list_v, list_f, list_c, 
                trackline_start_oppo - vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_end_oppo - vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_end_oppo - vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_start_oppo - vec_linecross*i/10 + vec_x * width_half_white_line,
                color=COLOR_TRACK_INTERVAL
            )
            
            add_a_rec(
                list_v, list_f, list_c, 
                trackline_start_oppo - vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_end_oppo - vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_end_oppo - vec_linecross*(i+1)/10 - vec_x * width_half_white_line, 
                trackline_start_oppo - vec_linecross*(i+1)/10 - vec_x * width_half_white_line,
                color=COLOR_TRACK
            )
        for i in [num_track]:
            add_a_rec(
                list_v, list_f, list_c, 
                trackline_start_oppo - vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_end_oppo - vec_linecross*i/10 - vec_x * width_half_white_line, 
                trackline_end_oppo - vec_linecross*i/10 + vec_x * width_half_white_line, 
                trackline_start_oppo - vec_linecross*i/10 + vec_x * width_half_white_line,
                color=COLOR_TRACK_INTERVAL
            )


        #draw culve
        #near side
        for i in range(num_track):
            add_an_arc(
                list_v, list_f, list_c, color=COLOR_TRACK_INTERVAL,
                p1=trackline_start + vec_linecross*i/10 + vec_x * width_half_white_line,
                p2=trackline_start_oppo - vec_linecross*i/10 - vec_x * width_half_white_line,
                d=width_half_white_line*2,
                normal=ground[0:3] * -1
            )
            add_an_arc(
                list_v, list_f, list_c, color=COLOR_TRACK,
                p1=trackline_start + vec_linecross*i/10 - vec_x * width_half_white_line,
                p2=trackline_start_oppo - vec_linecross*i/10 + vec_x * width_half_white_line,
                d=np.linalg.norm(vec_linecross, ord=2)/10 - 2 * width_half_white_line,
                normal=ground[0:3] * -1
            )
        for i in [num_track]:
            add_an_arc(
                list_v, list_f, list_c, color=COLOR_TRACK_INTERVAL,
                p1=trackline_start + vec_linecross*i/10 + vec_x * width_half_white_line,
                p2=trackline_start_oppo - vec_linecross*i/10 - vec_x * width_half_white_line,
                d=width_half_white_line*2,
                normal=ground[0:3] * -1
            )

        #far side
        for i in range(num_track):
            add_an_arc(
                list_v, list_f, list_c, color=COLOR_TRACK_INTERVAL,
                p1=trackline_end_oppo - vec_linecross*i/10 - vec_x * width_half_white_line,
                p2=trackline_end + vec_linecross*i/10 + vec_x * width_half_white_line,
                d=width_half_white_line*2,
                normal=ground[0:3] * -1
            )
            add_an_arc(
                list_v, list_f, list_c, color=COLOR_TRACK,
                p1=trackline_end_oppo - vec_linecross*i/10 + vec_x * width_half_white_line,
                p2=trackline_end + vec_linecross*i/10 - vec_x * width_half_white_line,
                d=np.linalg.norm(vec_linecross, ord=2)/10 - 2 * width_half_white_line,
                normal=ground[0:3] * -1
            )
        for i in [num_track]:
            add_an_arc(
                list_v, list_f, list_c, color=COLOR_TRACK_INTERVAL,
                p1=trackline_end_oppo - vec_linecross*i/10 - vec_x * width_half_white_line,
                p2=trackline_end + vec_linecross*i/10 + vec_x * width_half_white_line,
                d=width_half_white_line*2,
                normal=ground[0:3] * -1
            )
            

            
        
    #get_field
    def get_field():
        '''
                          lineX
        point_start_std +-----------+ point_oppo_std
                        |           |
                        |           |
                        |           |
            linescriptY |           | 
                        |           |
                        |           |
                        |           |
                        +---- ------+ 
        '''

        step_sum = 0
        for i in range(26):
            p1 = point_start_std + (step_sum - steps_26[0]) * vec_y
            p2 = point_start_std + (step_sum + steps_26[i] - steps_26[0]) * vec_y
            p3 = point_oppo_std + (step_sum + steps_26[i] - steps_26[0]) * vec_y
            p4 = point_oppo_std + (step_sum  - steps_26[0]) * vec_y
            step_sum = step_sum + steps_26[i]

            #vec_x_tmp = point_oppo_std - point_start_std
            #vec_x_tmp = vec_x_tmp / np.linalg.norm(vec_x_tmp)
            #print(np.dot(vec_x, vec_y))



            if i % 2 == 0:
                #color = [46, 79, 57]
                color = [181, 241, 93]
                #color = [99, 126, 123]
                
            else:
                #color = [65, 164, 119]
                color = [134, 169, 83]
                #color = [97, 184, 181]

            if i % 2 == 0:
                color = [85, 92, 34]
            else:
                color = [225, 227, 106]

            add_a_rec(list_v, list_f, list_c, p1, p2, p3, p4, color=color)
            

    
    def get_foundation():
        '''   
        point4 +-----------+ point3
               |           |
               |           |
               |           |
               |           | 
               |           |
               |           |
               |           |
        point2 +---- ------+ point1
        reverse view (because:
                ^x
                |
                |
            ----+------>y
                |
                |
        )  
        '''
        stand_edge_1 = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['field_edge_y']), ground=ground, cam_in=camera)
        point_stand_std = stand_edge_1[:, 0]
        point_stand_std[1] = point_stand_std[1]
        point2_std = point_stand_std + (-100) * vec_y + (-100) * vec_x
        point4_std = point2_std + (240) * vec_y
        point3_std = point4_std + (300) * vec_x
        point1_std = point2_std + (300) * vec_x
        add_a_rec(list_v, list_f, list_c, point1_std, point2_std, point4_std, point3_std, color=[120, 120, 120])    
    
    get_track()
    get_field()
    get_foundation()

    verts = np.array(list_v, dtype=np.float32)
    faces = np.array(list_f, dtype=np.int32)
    colors = np.array(list_c, dtype=np.uint8)
    write_obj_color_int(verts, colors, faces, './playground1_00_landmark.obj')


def landmark_to_obj(landmark_image:Dict[str, np.ndarray], camera:np.ndarray, ground:np.ndarray):
    #dict_keys(['trackline_edge', 'vec_x', 'vec_y', 'stand_edge_1', 'field_edge_y', 'field_edge_x'])
    list_v = []
    list_f = []
    list_c = []
    
    #get_vec
    def get_direction_from_line(line:np.ndarray):
        vec = uv_to_xyz_via_ground(uv_to_uv1T(line), ground=ground, cam_in=camera)
        vec = vec[:, 1] - vec[:, 0]
        vec = vec / np.linalg.norm(vec, ord=2)
        return vec
    vec_x = get_direction_from_line(landmark_image['vec_x'])
    vec_y = get_direction_from_line(landmark_image['vec_y'])

    #get_track
    def get_track():
        '''
         point4 +----+ point3
                |    |
                |    |
                |    |
          line2 |    | line1
                |    |
                |    |
                |    |
         point2 +----+ point1
        '''
        line1 = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['trackline_edge']), ground=ground, cam_in=camera)
        line2 = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['trackline_edge_2']), ground=ground, cam_in=camera)
        point1 = line1[:, 0]
        point2 = line2[:, 0]
        point3 = line1[:, 1]
        point4 = line2[:, 1]

        point1_std = point1

        vec12 = point2 - point1
        vec12_std = vec_x * np.dot(vec12, vec_x)
        point2_std = point1 + vec12_std

        vec13 = point3 - point1
        vec13_std = vec_y * np.dot(vec13, vec_y)
        point3_std = point1 + vec13_std

        point4_std = point2_std + vec13_std
    
        num_track = 10
        width = np.linalg.norm(vec12_std, ord=2) / num_track
        
        half_width_white_line = 0.04

        for i in range(num_track):
            
            add_a_rec(
                list_v, list_f, list_c, 
                point1_std + vec12_std*i/10 + vec_x * half_width_white_line, 
                point3_std + vec12_std*i/10 + vec_x * half_width_white_line, 
                point3_std + vec12_std*i/10 - vec_x * half_width_white_line, 
                point1_std + vec12_std*i/10 - vec_x * half_width_white_line,
                color=[255, 255, 255]
            )
            
            add_a_rec(
                list_v, list_f, list_c, 
                point1_std + vec12_std*i/10 - vec_x * half_width_white_line, 
                point3_std + vec12_std*i/10 - vec_x * half_width_white_line, 
                point3_std + vec12_std*(i+1)/10 + vec_x * half_width_white_line, 
                point1_std + vec12_std*(i+1)/10 + vec_x * half_width_white_line,
                color=[64, 64, 212]
            )
        for i in [num_track]:
            
            add_a_rec(
                list_v, list_f, list_c, 
                point1_std + vec12_std*i/10 + vec_x * half_width_white_line, 
                point3_std + vec12_std*i/10 + vec_x * half_width_white_line, 
                point3_std + vec12_std*i/10 - vec_x * half_width_white_line, 
                point1_std + vec12_std*i/10 - vec_x * half_width_white_line,
                color=[255, 255, 255]
            )
            
        
    #get_field
    def get_field():
        '''
                          lineX
        point_start_std +-----------+ point_oppo_std
                        |           |
                        |           |
                        |           |
            linescriptY |           | 
                        |           |
                        |           |
                        |           |
                        +---- ------+ 
        '''
        field_edge_x = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['field_edge_x']), ground=ground, cam_in=camera)
        field_edge_y = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['field_edge_y']), ground=ground, cam_in=camera)

        point_start_std = field_edge_y[:, 0]
        point_oppo = field_edge_x[:, 1]
        
        vec_st_op = point_oppo - point_start_std
        vec_st_op_std = vec_x * np.dot(vec_st_op, vec_x)
        point_oppo_std = point_start_std + vec_st_op_std

        num_rec = field_edge_y.shape[1] - 2
        step_sum = 0
        for i in range(num_rec):
            vec_step = field_edge_y[:, i + 1] - field_edge_y[:, i]
            vec_step_std = np.dot(vec_step, vec_y)
            p1 = point_start_std + step_sum * vec_y
            p2 = point_start_std + (step_sum + vec_step_std) * vec_y
            p3 = point_oppo_std + (step_sum + vec_step_std) * vec_y
            p4 = point_oppo_std + step_sum * vec_y
            step_sum = step_sum + vec_step_std
            if i % 2 == 0:
                color = [46, 79, 57]
            else:
                color = [65, 164, 119]

            add_a_rec(list_v, list_f, list_c, p1, p2, p3, p4, color=color)

    #get_stand
    def get_stand():
        '''
                   16m       
        point4 +-----------+ point3
               |           |
               |           |
               |           |
               |           | 
               |           |
               |           |
               |           |
        point2 +---- ------+ point1
        
        '''
        stand_edge_1 = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['stand_edge_1']), ground=ground, cam_in=camera)
        line1 = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['trackline_edge']), ground=ground, cam_in=camera)
        point3 = line1[:, 1]
        point1_std = stand_edge_1[:, 0]
        vec_13 = point3 - point1_std
        vec_13_std = vec_y * np.dot(vec_13, vec_y)
        point3_std = point1_std + vec_13_std
        point4_std = point3_std + (-16) * vec_x
        point2_std = point1_std + (-16) * vec_x


        add_a_rec(list_v, list_f, list_c, point1_std, point3_std, point4_std, point2_std, color=[74, 86, 99])
    
    def get_foundation():
        '''   
        point4 +-----------+ point3
               |           |
               |           |
               |           |
               |           | 
               |   p_stand |
               |---+       |
               |   |       |
        point2 +---- ------+ point1
        '''
        stand_edge_1 = uv_to_xyz_via_ground(uv_to_uv1T(landmark_image['stand_edge_1']), ground=ground, cam_in=camera)
        point_stand_std = stand_edge_1[:, 0]
        point_stand_std[1] = point_stand_std[1] + 0.001
        point2_std = point_stand_std + (-15) * vec_y + (-16) * vec_x
        point4_std = point2_std + (150) * vec_y
        point3_std = point4_std + (120) * vec_x
        point1_std = point2_std + (120) * vec_x

        add_a_rec(list_v, list_f, list_c, point1_std, point3_std, point4_std, point2_std, color=[75, 75, 75])    
    
    get_track()
    get_field()
    get_stand()
    get_foundation()

    verts = np.array(list_v, dtype=np.float32)
    faces = np.array(list_f, dtype=np.int32)
    colors = np.array(list_c, dtype=np.uint8)
    write_obj_color_int(verts, colors, faces, './1.obj')