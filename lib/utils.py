import json
import os
import numpy as np
import cv2
import pickle
import yaml
from scipy.spatial import Delaunay


def write_json(savename, data):
    with open(savename, 'w') as wf:
        json.dump(data, wf)


def read_json(json_name):
    with open(json_name, 'r') as rf:
        return json.load(rf)


def read_pkl(pkl_name):
    with open(pkl_name, 'rb') as rf:
        return pickle.load(rf)


def write_pkl(pkl_name, data):
    with open(pkl_name, 'wb') as wf:
        pickle.dump(data, wf)


def load_ground_cam(path_root):
    scene_type_list = os.listdir(path_root)
    param_dict = {}
    for scene_type in scene_type_list:
        ground = np.load(os.path.join(path_root, scene_type, 'ground.npy'))
        cam = np.load(os.path.join(path_root, scene_type, 'cam_para.npy'))
        param_dict[scene_type] = {'ground': ground, 'cam': cam}
    return param_dict

def update_yml(yml_path, modify):
    if modify is None:
        return
    with open(yml_path, 'r') as rf:
        data=yaml.load(rf, Loader=yaml.FullLoader)
    for key1 in modify:
        if key1 == 'sample_prob':
            data[key1]=modify[key1]
        else:
            for key2 in modify[key1]:
                data[key1][key2]=modify[key1][key2]
    with open(yml_path, 'w') as wf:
        yaml.dump(data, wf)


def save_obj(verts, faces, obj_mesh_name='mesh.obj'):
    #print('Saving:',obj_mesh_name)
    with open(obj_mesh_name, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

def save_multiperson_obj(verts, faces, obj_name):
    '''
    input:
        verts: numpy (N,6890,3)
        save_scene_path: for save
    '''

    num_of_person = verts.shape[0]
    with open(obj_name, 'w') as fp:
        for  batch_idx in range(num_of_person):
            verts_i = verts[batch_idx]
            for v in verts_i:
                fp.write( 'v %.7f %.7f %.7f\n' % ( v[0], v[1], v[2]) )
        for i in range(num_of_person):
            for f in faces: # Faces are 1-based, not 0-based in obj files
                f = f + 6890*i
                fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

def greedyTriangulationByY(points):
    points=points[:,[0,2]]
    tri = Delaunay(points)
    return tri.simplices

def AddgreedyBound(all_foot_3d, input_params=None, ground=None):
    if ground is None:
        ground = np.array(input_params['ground'])
    all_foot_3d=all_foot_3d.reshape(-1, 3)
    # 类似四边形状
    x_all, z_all = all_foot_3d[:, 0], all_foot_3d[:, 2]
    x_min, x_max = x_all.min(), x_all.max()
    z_min, z_max = z_all.min(), z_all.max()
    
    delta=2
    x_min-=delta
    x_max+=delta
    z_min-=delta
    z_max+=delta

    points_xz = np.array([[x_min, z_min], [x_max, z_min], [x_min, z_max], [x_max, z_max]]).astype(int)
    points_x=points_xz[:, 0]
    points_z=points_xz[:, 1]
    points_y = -(ground[0]*points_x + ground[2]*points_z+ground[3])/ground[1]
    #print('points on ground', points_x*ground[0]+points_y*ground[1]+points_z*ground[2]+ground[3])
    points=np.stack([points_x, points_y, points_z], axis=-1)
    new_all_foot_3d=np.concatenate([points, all_foot_3d], axis=0)
    #print('foot3d on ground', all_foot_3d[:,0]*ground[0]+all_foot_3d[:,1]*ground[1]+all_foot_3d[:,2]*ground[2]+ground[3])
    return points #new_all_foot_3d


def kps2bbox(kps):
    # kps(w,h): m, 2
    w0=kps[:, 0].min()
    w1=kps[:, 0].max()
    h0=kps[:, 1].min()
    h1=kps[:, 1].max()
    return [w0, h0, w1, h1]

def get_valid_bbox(bbox, pj2d):
    patch_w1,patch_h1,patch_w2,patch_h2=bbox
    kps_w1, kps_h1, kps_w2, kps_h2=kps2bbox(pj2d)
    res_w1=max([kps_w1, patch_w1])
    res_w2=min([kps_w2, patch_w2])
    res_h1=max([kps_h1, patch_h1])
    res_h2=min([kps_h2, patch_h2])
    return [int(res_w1), int(res_h1), int(res_w2), int(res_h2)], [int(kps_w1), int(kps_h1), int(kps_w2), int(kps_h2)]


