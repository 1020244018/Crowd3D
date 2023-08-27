from test_configs import *
from test_constants import *
import os, time
import subprocess
import copy
from lib.utils import update_yml
from lib.quantative import compute_quantative_for_submit
from global_setting import *
from tqdm import tqdm
import argparse


import argparse

parser=argparse.ArgumentParser(description='inference single image')
parser.add_argument('--use_pre_optim', action='store_true',  help='If True, use our optimized weights and ground params')
parser.add_argument('--optim_batch_size', type=int, default=64, help='the batch_size in optim process')
parser.add_argument('--optim_epoch', type=int, default=160)
parser.add_argument('--inference_batch_size', type=int, default=128, help='the batch_size in inference process')
args=parser.parse_args()
use_pre_optim=args.use_pre_optim # dirct use the ground and optimed-cpts we provided 

mid_root=os.path.join(project_root, 'test', 'test_mid')
scene_image_root=os.path.join(project_root, 'data/largecrowd/images/test')
ground_root=os.path.join(project_root, 'test', 'test_mid', 'ground_mid', 'ground')
# ground_root=os.path.join(project_root, 'params', 'pre_grounds', 'ground_info_predict_1106')

save_folder=os.path.join(project_root, 'test', 'test_res', 'infer')
submit_folder=os.path.join(project_root, 'test', 'test_res', 'submit')

base_cpt=os.path.join(crowd3dnet_root, 'pretrained/base_69.pkl')
optimized_folder=os.path.join(crowd3dnet_root, 'pretrained', 'optimized')

optim_epoch=args.optim_epoch
optim_batch_size=args.optim_batch_size # Setting too large is not recommended unless there is sufficient data volume
val_batch_size=args.inference_batch_size
time_dict={}

run_dict={
    'finetune_flag': True,
    'inference_flag': True,
    'quantative_flag': True,
    'visual_mesh_flag': False # Need to set the scene_image
}

if use_pre_optim:
    print('use the pre-optimized.')
    ground_root=os.path.join(project_root, 'params', 'pre_grounds', 'ground_info_predict_1106')

## 1. finetune
finetune_flag=run_dict['finetune_flag']
optim_tag_post=os.path.basename(ground_root)
if finetune_flag and not use_pre_optim:
    print('4. Start Finetune')
    start_time=time.time()
    for scene_type in eval_scenes:
        yml_file=os.path.join(crowd3dnet_root, 'configs', 'optim', 'optim.yml')
        modify_dict= {
            'ARGS': {
                'finetune_mode': 'layer1_gc',
                'tab': 'optim'
            }
        }
        modify_dict['ARGS']['tab']+='_'+scene_type+'_'+optim_tag_post
        modify_dict['ARGS']['model_path']=base_cpt
        modify_dict['ARGS']['test_train_ground_path']=os.path.join(ground_root, scene_type)
        cur_frame_mid=os.path.join(mid_root, scene_type, select_frame_dict[scene_type][0][:-4])
        modify_dict['ARGS']['test_train_image_path']=os.path.join(cur_frame_mid, 'part_images_scale')
        modify_dict['ARGS']['test_train_anno_path']=os.path.join(cur_frame_mid, 'optim_annots_scale.pkl')
        modify_dict['ARGS']['batch_size']=optim_batch_size
        modify_dict['ARGS']['epoch']=optim_epoch
        modify_dict['ARGS']['dataset']='test_train_scale'
        modify_dict['sample_prob']={'test_train_scale':1}
        update_yml(yml_file, modify_dict)
        command='python -m crowd3d.train --configs_yml='+yml_file
        subprocess.run(command, shell=True, cwd=crowd3dnet_root)
    end_time=time.time()
    time_dict['finetune']=end_time-start_time

epoch_list=[max([optim_epoch-1, 0])] # [59, 79, 99, 119, 139, 159]

## 2. inference
inference_flag=run_dict['inference_flag']
if inference_flag:
    print('5. Start Inference')
    start_time=time.time()
    if use_pre_optim:
        inference_folder=os.path.join(save_folder, 'use_pre_optim')
        for scene_type in eval_scenes:
            checkpoint_path=os.path.join(optimized_folder, scene_type, 'epoch_159.pkl')
            output_dir=os.path.join(inference_folder, scene_type)
            inputs_path=mid_root
            base_command = 'python -m crowd3d.predict.inference_largecrowd'
            command = base_command + ' --configs_yml=configs/inference/eval_largecrowd.yml'+' --model_path ' + checkpoint_path + ' --inputs ' + inputs_path + ' --ground_cam_root '+ ground_root + ' --output_dir ' + output_dir +' --scene_type '+ scene_type + ' --val_batch_size ' + str(val_batch_size)
            subprocess.run(command, shell=True, cwd=crowd3dnet_root)
    else:
        cpt_tag=''
        for epoch in epoch_list: #[0, 19, 39, 59, 79, 99, 119, 139, 159, 179, 199]:
            inference_folder=os.path.join(save_folder,  optim_tag_post+'_epoch_'+str(epoch))
            for scene_type in eval_scenes:
                cpt_tag='optim_'+scene_type+'_'+optim_tag_post
                cpt_folder='hrnet_cm64_'+ cpt_tag + '_on_gpu0,1_val'
                cpt_name='hrnet_cm64_' + cpt_tag + '_epoch_'+str(epoch)+'.pkl'
                checkpoint_path=os.path.join(crowd3dnet_root, 'checkpoints', cpt_folder, cpt_name)
                if epoch <= 0:
                    checkpoint_path=base_cpt
                output_dir=os.path.join(inference_folder, scene_type)
                inputs_path=mid_root
                base_command = 'python -m crowd3d.predict.inference_largecrowd'
                command = base_command + ' --configs_yml=configs/inference/eval_largecrowd.yml'+' --model_path ' + checkpoint_path + ' --inputs ' + inputs_path + ' --ground_cam_root '+ ground_root + ' --output_dir ' + output_dir +' --scene_type '+ scene_type + ' --val_batch_size ' + str(val_batch_size)
                subprocess.run(command, shell=True, cwd=crowd3dnet_root)
    end_time=time.time()  
    time_dict['inference']=end_time-start_time

## 3. quantitation
quantative_flag=run_dict['quantative_flag']
quantative_save_root='quantative_results'
if quantative_flag:
    print('6. Generate submitted JSON')
    start_time=time.time()
    if use_pre_optim:
        inference_folder=os.path.join(save_folder, 'use_pre_optim')
        cur_submit_folder=os.path.join(submit_folder, 'use_pre_optim')
        compute_quantative_for_submit(inference_folder, scene_image_root, ground_root,scene_types=eval_scenes, mid_root=mid_root, save_json_folder=cur_submit_folder, merge_mode='tc')
    else:
        for epoch in epoch_list: # [0, 19, 39, 59, 79, 99, 119, 139, 159, 179, 199]: #[0, 9, 19, 29, 39, 49, 59, 69]:  
            inference_folder=os.path.join(save_folder,  optim_tag_post+'_epoch_'+str(epoch))
            cur_submit_folder=os.path.join(submit_folder, optim_tag_post+'_epoch_'+str(epoch))
            compute_quantative_for_submit(inference_folder, scene_image_root, ground_root,scene_types=eval_scenes, mid_root=mid_root, save_json_folder=cur_submit_folder, merge_mode='tc')
        
    end_time=time.time()
    time_dict['quantative']=end_time-start_time



## 4. visual mesh on scene image
visual_mesh_flag=run_dict['visual_mesh_flag']
if visual_mesh_flag:
    start_time=time.time()
    
    if use_pre_optim:
        inference_folder=os.path.join(save_folder, 'use_pre_optim')
    else:
        epoch=epoch_list[-1]
        inference_folder=os.path.join(save_folder,  optim_tag_post+'_epoch_'+str(epoch))
    
    
    render=True
    with_hvip=False
    save_mesh=False
    save_mesh_mode='all' # one of ['all', 'single', 'multi']
    scene_image='playground1_00_001740.jpg' # 'stadiumEntrance_00_002300.jpg'  # 'playground1_00_006400.jpg'


    scene_type=scene_image.split('_')[0] + '_' +scene_image.split('_')[1]
    result_root=os.path.join(inference_folder, scene_type)
    save_folder=os.path.join(inference_folder, scene_type, 'visual', scene_image.replace('.jpg',''))
    scene_image_path=os.path.join(scene_image_root, scene_type, scene_image)
    scene_mid_root=os.path.join(mid_root, scene_type, scene_image.replace('.jpg', ''))
    ground_cam_path=os.path.join(ground_root, scene_type)

    base_command = 'python render/render_scene.py '\
        + '--result_path ' + os.path.join(result_root, 'all_result.pkl') + ' '\
        + '--scene_image_path ' + scene_image_path + ' '\
        + '--save_root ' + save_folder + ' '\
        + '--mid_root ' + scene_mid_root + ' '\
        + '--ground_cam_path ' + ground_cam_path + ' '\
        + '--save_mesh_mode ' + save_mesh_mode

    action_flag_list=[render, with_hvip, save_mesh]
    action_list=['is_render', 'with_hvip', 'is_save_mesh']
    for flag, action in zip(action_flag_list, action_list):
        if flag:
            base_command=base_command + ' --' +action
    command=base_command
    print(command)
    subprocess.run(command, shell=True)


    
    time_dict['visual_mesh']=time.time()-start_time

print(time_dict)