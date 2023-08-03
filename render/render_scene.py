import os
import numpy as np
import cv2
import torch
import argparse
import sys
project_path=os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)
from lib.merge_utils import merge_return_cid, extract_results_after_merged
from lib.utils import read_pkl, save_obj, greedyTriangulationByY, AddgreedyBound, save_multiperson_obj, get_valid_bbox
from lib.render_utils import get_rotate_X, get_rotate_Y, get_rotate_Z, rotate_RT, trans_RT, render, convert_trimesh
from lib.smpl import SMPL
from global_setting import *

os.environ['PYOPENGL_PLATFORM']='egl'

parser=argparse.ArgumentParser(description='render mesh on scene image')
parser.add_argument('--result_path', type=str, default='')
parser.add_argument('--scene_image_path', type=str, default='')
parser.add_argument('--save_root', type=str, default='')
parser.add_argument('--mid_root', type=str, default='')
parser.add_argument('--ground_cam_path', type=str, default='')
parser.add_argument('--is_render', action='store_true')
parser.add_argument('--is_save_mesh', action='store_true')
parser.add_argument('--save_mesh_mode', type=str, default='multi') # 'multi' or 'single' or 'all'
parser.add_argument('--with_hvip', action='store_true')
parser.add_argument('--correct_smpl_scale_path', type=str, default='')
parser.add_argument('--mask_path', type=str, default='')
args=parser.parse_args()




render_flag=args.is_render
extract_mesh_flag=args.is_save_mesh

mask_path=args.mask_path
ground_cam_path=args.ground_cam_path
scene_image_path = args.scene_image_path
mid_result_path = args.mid_root
scene_name = os.path.basename(scene_image_path)
position_path = os.path.join(mid_result_path, 'position.npy')
row_col_path = os.path.join(mid_result_path, 'row_col.npy')
name_pre = scene_name.replace('.jpg', '')
scene_type=scene_name.split('_')[0] + '_' +scene_name.split('_')[1]
ground=np.load(os.path.join(ground_cam_path, 'ground.npy'))
camK=np.load(os.path.join(ground_cam_path, 'cam_para.npy'))
save_root=args.save_root
os.makedirs(save_root, exist_ok=True)

all_result = read_pkl(args.result_path)
all_trans=[]
smpl_n=SMPL(smpl_model_root, J_reg_extra9_path=os.path.join(smpl_model_root, 'J_regressor_extra.npy'),  gender='neutral')
correct_smpl_scale=1.
if args.correct_smpl_scale_path != '':
    correct_smpl_scale_dict=read_pkl(args.correct_smpl_scale_path)
    correct_smpl_scale=correct_smpl_scale_dict[scene_type]


# extract_mesh
if extract_mesh_flag:
    extract_mesh_save=os.path.join(save_root, 'extract_mesh')
    os.makedirs(extract_mesh_save, exist_ok=True)
    if args.save_mesh_mode in ['single', 'all']:
        everyone_path=os.path.join(extract_mesh_save, 'everyone_by_pathch')
        os.makedirs(everyone_path, exist_ok=True)

final_center_points, final_center_points_xy, total_cps_num, cps2id_dict, x2row_dict, y2col_dict = merge_return_cid(
    position_path, row_col_path,
    name_pre, all_result, mask_path=mask_path, mode='tc')
merge_result=extract_results_after_merged(final_center_points,final_center_points_xy,cps2id_dict, all_result,x2row_dict,y2col_dict, name_pre)


scene_image = cv2.imread(scene_image_path)
scene_h, scene_w, _ = scene_image.shape 





# # hvip2d
# if args.with_hvip:
#     all_hvip2d=[]
#     bg_image=scene_image.copy()
#     h_b, w_b, _ = bg_image.shape
#     for key in merge_result:
#         cur_data=merge_result[key]
#         for p_id in range(len(cur_data)):
#             cur_hvip2d=cur_data[p_id]['hvip2d_in_scene']
#             all_hvip2d.append(cur_hvip2d)
#             cur_color=np.random.rand(3)*255
#             cv2.circle(bg_image, (int(cur_hvip2d[0]), int(cur_hvip2d[1])), 10, cur_color, thickness=-1)
#     hvip_save_path=os.path.join(save_root, name_pre+'_hvip.jpg')

#     cv2.imwrite(hvip_save_path, bg_image)
#     print('with_hvip. save as %s' %hvip_save_path)
#     del bg_image

faces = smpl_n.faces
torso_index=[16, 17, 45, 46]

extract_all_verts_cam=[]
extract_all_hvip3d=[]
extract_mesh_by_patch={}
extract_hvip3d_by_patch={}

for key in merge_result:
    cur_data=merge_result[key]
    cur_poses, cur_betas, cur_trans=[], [], []
    extract_mesh_by_patch[key]=[]
    extract_hvip3d_by_patch[key]=[]
    for p_id in range(len(cur_data)):
        cur_poses.append(cur_data[p_id]['pose'])
        cur_betas.append(cur_data[p_id]['betas'])
        cur_trans.append(cur_data[p_id]['final_trans'])
        extract_all_hvip3d.append(cur_data[p_id]['hvip3d'])
        extract_hvip3d_by_patch[key].append(cur_data[p_id]['hvip3d'])
    cur_poses, cur_betas, cur_trans = np.array(cur_poses), np.array(cur_betas), np.array(cur_trans)

    cur_poses=torch.from_numpy(cur_poses).float()
    cur_betas=torch.from_numpy(cur_betas).float()
    smpl_out=smpl_n(poses=cur_poses, betas=cur_betas)
    verts, j3ds=smpl_out['verts'].numpy(), smpl_out['j3d'].numpy()
    trans=np.expand_dims(cur_trans, 1)
    verts_cam, j3ds_cam = verts + trans, j3ds + trans
    torso_root_cam=np.expand_dims(j3ds_cam[:, torso_index].mean(1), 1)
    torso_root_smpl=np.expand_dims(j3ds[:, torso_index].mean(1), 1)
    verts_cam = (verts - torso_root_smpl) * correct_smpl_scale + torso_root_cam
    for p_id,v in enumerate(verts_cam):
        extract_all_verts_cam.append(v)
        extract_mesh_by_patch[key].append(v)


if render_flag:
    visible_weight = 0.8
    trimesh_list=convert_trimesh(extract_all_verts_cam, faces)
    base_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0],
    ])

    camera_pose = base_pose
    light_pose = rotate_RT(base_pose, get_rotate_X(3.5 * np.pi / 2))

    f = camK[0, 0]
    cx = camK[0, 2]
    cy = camK[1, 2]

    render_image, depth = render(trimesh_list, f, cx, cy, camera_pose, light_pose, image_shape=[scene_image.shape[0], scene_image.shape[1]])

    mask=np.zeros((render_image.shape[0], render_image.shape[1]))
    temp=render_image[:,:,0]+render_image[:,:,1]+render_image[:,:,2]
    mask[temp>0]=1
    mask=np.expand_dims(mask, -1)

    render_scene_image=render_image * mask * visible_weight + (1-mask*visible_weight)* scene_image
    render_mesh_path=os.path.join(save_root, name_pre+'_result.jpg')
    cv2.imwrite(render_mesh_path, render_scene_image)
    print('render mesh. save as %s' %render_mesh_path)

    if args.with_hvip:
        all_hvip2d=[]
        bg_image=render_scene_image
        h_b, w_b, _ = bg_image.shape
        for key in merge_result:
            info=key.split('_')
            cur_size=int(info[-2]) - int(info[-4])
            cur_data=merge_result[key]
            for p_id in range(len(cur_data)):
                cur_hvip2d=cur_data[p_id]['hvip2d_in_scene']
                all_hvip2d.append(cur_hvip2d)
                cur_color= np.array([254., 254., 0.])# np.random.rand(3)*255
                circle_size=np.max([int(cur_size*0.005), 2])
                cv2.circle(bg_image, (int(cur_hvip2d[0]), int(cur_hvip2d[1])), circle_size, cur_color, thickness=2)
        lp_save_path=os.path.join(save_root, name_pre+'_with_hvip.jpg')

        cv2.imwrite(lp_save_path, bg_image)
        print('with_hvip. save as %s' %lp_save_path)
        del bg_image

if extract_mesh_flag:
    if args.save_mesh_mode in ['single', 'all']:
        for patch_name in extract_mesh_by_patch:
            verts_list=extract_mesh_by_patch[patch_name]
            patch_name_pre=patch_name.replace('.jpg','')
            cur_save_folder=os.path.join(everyone_path, patch_name_pre)
            os.makedirs(cur_save_folder, exist_ok=True)
            for p_id,v in enumerate(verts_list):
                save_obj(v, faces, obj_mesh_name=os.path.join(cur_save_folder, str(p_id)+'.obj'))
            # ground
            extract_patch_hvip3d=np.array(extract_hvip3d_by_patch[patch_name]).reshape(-1, 3)
            new_patch_hvip_3d=AddgreedyBound(extract_patch_hvip3d, ground=ground)
            new_faces = greedyTriangulationByY(new_patch_hvip_3d)
            save_obj(new_patch_hvip_3d, new_faces, obj_mesh_name=os.path.join(cur_save_folder, 'patch_ground.obj'))

    # ground
    extract_all_hvip3d=np.array(extract_all_hvip3d).reshape(-1, 3)
    new_all_hvip_3d=AddgreedyBound(extract_all_hvip3d, ground=ground)
    new_faces = greedyTriangulationByY(new_all_hvip_3d)
    save_obj(new_all_hvip_3d, new_faces, obj_mesh_name=os.path.join(extract_mesh_save, 'ground.obj'))

    if args.save_mesh_mode in ['multi', 'all']:
        all_v=np.array(extract_all_verts_cam)
        save_multiperson_obj(all_v, faces, obj_name=os.path.join(extract_mesh_save, name_pre+'.obj'))
        all_v=np.array(extract_all_verts_cam)[:10,:, :]
        save_multiperson_obj(all_v, faces, obj_name=os.path.join(extract_mesh_save, 'small_check.obj'))
    print('extract mesh. type: %s. save in %s' % (args.save_mesh_mode, extract_mesh_save))



