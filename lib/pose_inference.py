import subprocess
import os


def pose_inference(input_path, save_path, cwd, save_image=False, vis_fast=False, use_scale_blocks=False):
    base_command = 'python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    command = base_command
    if save_image:
        command = command + ' --save_img'
    command = command + ' --indir ' + input_path + ' --outdir ' + save_path
    if vis_fast:
        command = command + ' --vis_fast'
    subprocess.run(command, shell=True, cwd=cwd)
    if use_scale_blocks: # 'alphapose-results.json' -> 'alphapose-results-scale.json'
        filename=os.path.join(save_path, 'alphapose-results.json')
        updatename=os.path.join(save_path, 'alphapose-results-scale.json')
        os.rename(filename, updatename)
    else:
        filename=os.path.join(save_path, 'alphapose-results.json')
        updatename=os.path.join(save_path, 'alphapose-results-origin.json')
        os.rename(filename, updatename)
        
