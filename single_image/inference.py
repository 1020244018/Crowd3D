import os
import numpy as np
from single_image_configs import *
import subprocess
import copy

from lib.utils import update_yml
from global_setting import *
import argparse

parser=argparse.ArgumentParser(description='inference single image')
parser.add_argument('--scene_image_path', type=str, default='')
parser.add_argument('--save_mid_folder', type=str, default='')
parser.add_argument('--save_res_folder', type=str, default='')

parser.add_argument('--optim_batch_size', type=int, default=64, help='the batch_size in optim process')
parser.add_argument('--optim_epoch', type=int, default=100)
parser.add_argument('--inference_batch_size', type=int, default=128, help='the batch_size in inference process')

parser.add_argument('--render_with_hvip', action='store_true')
parser.add_argument('--save_mesh', action='store_true',  help='save the crowd meshes as obj')
parser.add_argument('--save_mesh_mode', type=str, default='all', help='single, multi, all')

args=parser.parse_args()

run_dict={
    'finetune_flag': True,
    'inference_flag': True,
    'visual_mesh_flag': True
    }

scene_image_path=args.scene_image_path
save_mid_folder=args.save_mid_folder
save_res_folder=args.save_res_folder
optim_batch_size=args.optim_batch_size # Setting too large is not recommended unless there is sufficient data volume
optim_epoch=args.optim_epoch
inference_batch_size=args.inference_batch_size

if args.scene_image_path == '':
    print('Please input the scene_image_path')
    exit()
elif '.jpg' not in args.scene_image_path and '.png' not in args.scene_image_path:
    print('The image need has .jpg suffix')
    exit()
if save_mid_folder!='' and not os.path.exists(save_mid_folder):
    print('Can not find the folder %s' %save_mid_folder)
    exit()
if save_mid_folder=='':
    name_pre=os.path.basename(scene_image_path).replace('.jpg', '').replace('.png', '')
    save_mid_folder=os.path.join(project_root, 'single_image', 'single_image_mid', name_pre)
if save_res_folder=='':
    save_res_folder=os.path.join(project_root, 'single_image', 'single_image_res')

ground_folder=os.path.join(save_mid_folder, 'ground_mid', 'ground')
base_cpt=os.path.join(crowd3dnet_root, 'pretrained/base_69.pkl')
os.makedirs(save_res_folder, exist_ok=True)



##  finetune
if run_dict['finetune_flag']:
    print('4. finetune')
    yml_file=os.path.join(crowd3dnet_root, 'configs', 'optim', 'optim.yml')
    modify_dict= {
        'ARGS': {
            'finetune_mode': 'layer1_gc',
            'tab': 'optim'
        }
    }
    
    yml_file=os.path.join(crowd3dnet_root, 'configs', 'optim', 'optim_single_image.yml')
    modify_dict=copy.deepcopy(modify_dict)
    modify_dict['ARGS']['tab']=name_pre
    modify_dict['ARGS']['model_path']=base_cpt
    modify_dict['ARGS']['test_train_ground_path']=ground_folder
    modify_dict['ARGS']['test_train_image_path']=os.path.join(os.path.join(save_mid_folder, 'part_images_scale'))
    modify_dict['ARGS']['test_train_anno_path']=os.path.join(save_mid_folder, 'optim_annots_scale.pkl')
    modify_dict['ARGS']['batch_size']=optim_batch_size
    modify_dict['ARGS']['epoch']=optim_epoch
    modify_dict['ARGS']['dataset']='test_train_single_image'
    modify_dict['sample_prob']={'test_train_single_image':1}
    update_yml(yml_file, modify_dict)
    command='python -m crowd3d.train --configs_yml='+yml_file
    subprocess.run(command, shell=True, cwd=crowd3dnet_root)

##  inference
epoch=optim_epoch - 1
visul_inference=False
output_dir=os.path.join(project_root, save_res_folder, name_pre, 'epoch_'+str(epoch))
have_people_image_json=os.path.join(save_mid_folder, 'have_people_image_name.json') # or 'no' to cancel
if run_dict['inference_flag']:
    print('5. inference')
    inference_yml='single_scene_image.yml'
    inference_yml_path=os.path.join(crowd3dnet_root, 'configs', 'inference', inference_yml)
    update_yml(inference_yml_path, {'ARGS':{'save_visualization_on_img': visul_inference}})
    
    cpt_folder='hrnet_cm64_'+ name_pre + '_on_gpu0,1_val'
    cpt_name='hrnet_cm64_' + name_pre + '_epoch_'+str(epoch)+'.pkl'
    checkpoint_path=os.path.join(crowd3dnet_root, 'checkpoints', cpt_folder, cpt_name)
    if epoch <= 0:
        checkpoint_path=os.path.join(crowd3dnet_root, base_cpt)
    
    inputs_path=os.path.join(os.path.join(save_mid_folder, 'part_images_scale'))
    base_command = 'python -m crowd3d.predict.inference_single_scene_image'
    command = base_command + ' --configs_yml=configs/inference/'+inference_yml +' --model_path ' + checkpoint_path + ' --inputs ' + inputs_path + ' --ground_cam_path '+ ground_folder + ' --output_dir ' + output_dir + ' --val_batch_size ' + str(inference_batch_size) +' --have_people_image_json ' + have_people_image_json
    subprocess.run(command, shell=True, cwd=crowd3dnet_root)


##  visual mesh on scene image
if run_dict['visual_mesh_flag']:
    print('6. visual mesh')
    render=True
    with_hvip=args.render_with_hvip
    save_mesh=args.save_mesh
    
    
    save_mesh_mode=args.save_mesh_mode # one of ['all', 'single', 'multi']
    if save_mesh and save_mesh_mode not in ['all', 'single', 'multi']:
        save_mesh_mode='all'
    
    visual_folder=os.path.join(output_dir, 'visual')
    base_command = 'python render/render_scene.py '\
        + '--result_path ' + os.path.join(output_dir, 'all_result.pkl') + ' '\
        + '--scene_image_path ' + scene_image_path + ' '\
        + '--save_root ' + visual_folder + ' '\
        + '--mid_root ' + save_mid_folder + ' '\
        + '--ground_cam_path ' + ground_folder + ' '\
        + '--save_mesh_mode ' + save_mesh_mode
    
    action_flag_list=[render, with_hvip, save_mesh]
    action_list=['is_render', 'with_hvip', 'is_save_mesh']
    for flag, action in zip(action_flag_list, action_list):
        if flag:
            base_command=base_command + ' --' +action
    command=base_command
    subprocess.run(command, shell=True)
