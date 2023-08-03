import subprocess
import os
import numpy as np

def generate_ground(
        joint_path, scene_image_path, save_path, 
        fps_sample_num=-1,
        cwd='/mnt/wh/Crowd3D/estimate_ground_faster/', 
        close_vis=True,
        path_GT_cam:str or None=None
    ):
    base_command = 'python script/main.py'
    command = base_command

    #if path_GT_cam is None:
    #    path_save = '/media/panda_data/workforpandaperson/lock_you_on_the_ground_faster/results'
    #else:
    #    path_save = '/media/panda_data/workforpandaperson/lock_you_on_the_ground_faster/results_gt_cam'

    command = command + ' --joint_path ' + os.path.abspath(joint_path) + \
        ' --scene_image_path ' + os.path.abspath(scene_image_path) + \
        ' --save_path '+ os.path.abspath(save_path) + \
        ' --fps_sample_num ' + str(fps_sample_num)
    if close_vis:
        command = command + ' --close_vis'
    if path_GT_cam is not None:
        assert os.path.exists(path_GT_cam)
        command = command + ' --gt_cam_path ' + os.path.abspath(path_GT_cam)
    # print(command)
    subprocess.run(command, shell=True, cwd=cwd)



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




def trans_ground(ground, distant_to_ground=0.1):
    if ground[1] > 0:  # B must < 0
        ground=-ground
    
    mo=(ground[0]**2 + ground[1]**2 + ground[2]**2)**0.5
    D=distant_to_ground*mo+ground[3]
    ground[3]=D
    return ground

