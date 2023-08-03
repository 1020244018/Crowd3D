import os, sys
import argparse
import math
import numpy as np
import torch
import yaml
import logging
import time
import platform

currentfile = os.path.abspath(__file__)
code_dir = currentfile.replace('config.py', '')
project_dir = currentfile.replace('/crowd3d/lib/config.py', '')
source_dir = currentfile.replace('/lib/config.py', '')
root_dir = project_dir  # .replace(project_dir.split('/')[-1],'')

time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(int(round(time.time() * 1000)) / 1000))
yaml_timestamp = os.path.abspath(
    os.path.join(project_dir, "active_configs/active_context_{}.yaml".format(time_stamp).replace(":", "_")))

plt = platform.system()
if plt == "Windows":
    project_dir = currentfile.replace('\\crowd3d\\lib\\config.py', '')
    source_dir = currentfile.replace('\\lib\\config.py', '')
    root_dir = project_dir.replace(project_dir.split('\\')[-1], '')
    yaml_timestamp = os.path.abspath(
        os.path.join(source_dir + "active_configs\\active_context_{}.yaml".format(time_stamp.replace(":", "_"))))

public_params_root=os.path.join(project_dir, 'public_params/')
model_dir = os.path.join(public_params_root, 'model_data')
trained_model_dir = os.path.join(public_params_root, 'trained_models')

print("yaml_timestamp ", yaml_timestamp)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='Crowd3DNet')
    parser.add_argument('--tab', type=str, default='v1', help='additional tabs')
    parser.add_argument('--configs_yml', type=str, default='configs/v1.yml', help='settings')
    parser.add_argument('--inputs', type=str, help='path to inputs')
    parser.add_argument('--output_dir', type=str, help='path to save outputs')
    parser.add_argument('--project_dir', type=str, default=project_dir)

    parser.add_argument('--test_train', type=bool, default=False)
    parser.add_argument('--finetune_mode', type=str, default='')
    parser.add_argument('--ground_cam_path', type=str, default='')
    parser.add_argument('--mid_result_path', type=str, default='')
    parser.add_argument('--inference_file', type=str, default='')
    parser.add_argument('--scene_image_path', type=str, default='')
    parser.add_argument('--correct_smpl_scale', type=float, default=1.0)
    parser.add_argument('--ground_cam_root', type=str, default='')
    parser.add_argument('--scene_type', type=str, default='')
    parser.add_argument('--have_people_image_json', type=str, default='')

    mode_group = parser.add_argument_group(title='mode options')
    # mode settings
    mode_group.add_argument('--model_return_loss', type=bool, default=False,
                            help='wether return loss value from the model for balanced GPU memory usage')
    mode_group.add_argument('--model_version', type=int, default=1, help='model version')
    mode_group.add_argument('--multi_person', type=bool, default=True, help='whether to make Multi-person Recovery')
    mode_group.add_argument('--new_training', type=bool, default=False,
                            help='learning centermap only in first few iterations for stable training.')
    mode_group.add_argument('--perspective_proj', type=bool, default=False,
                            help='whether to use perspective projection, else use orthentic projection.')
    mode_group.add_argument('--new_training_epoch', type=int, default=-1,
                            help='learning centermap only in first few epochs for stable training.')

    train_group = parser.add_argument_group(title='training options')
    # basic training settings
    train_group.add_argument('--lr', help='lr', default=3e-4, type=float)
    train_group.add_argument('--adjust_lr_factor', type=float, default=0.1, help='factor for adjusting the lr')
    train_group.add_argument('--weight_decay', help='weight_decay', default=1e-6, type=float)
    train_group.add_argument('--epoch', type=int, default=120, help='training epochs')
    train_group.add_argument('--fine_tune', type=bool, default=True, help='whether to run online')
    train_group.add_argument('--gpu', default='0', help='gpus', type=str)
    train_group.add_argument('--batch_size', default=64, help='batch_size', type=int)
    train_group.add_argument('--input_size', default=512, type=int, help='size of input image')
    train_group.add_argument('--master_batch_size', default=-1, help='batch_size', type=int)
    train_group.add_argument('--nw', default=4, help='number of workers', type=int)
    train_group.add_argument('--optimizer_type', type=str, default='Adam', help='choice of optimizer')
    train_group.add_argument('--pretrain', type=str, default='simplebaseline',
                             help='imagenet or spin or simplebaseline')
    train_group.add_argument('--fix_backbone_training_scratch', type=bool, default=False,
                             help='whether to fix the backbone features if we train the model from scratch.')
    train_group.add_argument('--milestones', default='60,80', type=str, help='train milestones')

    model_group = parser.add_argument_group(title='model settings')
    # model settings
    model_group.add_argument('--backbone', type=str, default='hrnetv4', help='backbone model: resnet50 or hrnet')
    model_group.add_argument('--model_precision', type=str, default='fp16', help='the model precision: fp16/fp32')
    model_group.add_argument('--deconv_num', type=int, default=0)
    model_group.add_argument('--head_block_num', type=int, default=2, help='number of conv block in head')
    model_group.add_argument('--merge_smpl_camera_head', type=bool, default=False)
    model_group.add_argument('--use_coordmaps', type=bool, default=True, help='use the coordmaps')
    model_group.add_argument('--hrnet_pretrain', type=str,
                             default=os.path.join(trained_model_dir, 'pretrain_hrnet.pkl'))
    model_group.add_argument('--resnet_pretrain', type=str,
                             default=os.path.join(trained_model_dir, 'pretrain_resnet.pkl'))

    loss_group = parser.add_argument_group(title='loss options')
    # loss settings
    loss_group.add_argument('--loss_thresh', default=1000, type=float, help='max loss value for a single loss')
    loss_group.add_argument('--max_supervise_num', default=-1, type=int,
                            help='max person number supervised in each batch for stable GPU memory usage')
    loss_group.add_argument('--supervise_cam_params', type=bool, default=False)
    loss_group.add_argument('--match_preds_to_gts_for_supervision', type=bool, default=False,
                            help='whether to match preds to gts for supervision')
    loss_group.add_argument('--matching_mode', type=str, default='all', help='all | random_one | ')
    loss_group.add_argument('--supervise_global_rot', type=bool, default=False,
                            help='whether supervise the global rotation of the estimated SMPL model')
    loss_group.add_argument('--HMloss_type', type=str, default='MSE',
                            help='supervision for 2D pose heatmap: MSE or focal loss')

    eval_group = parser.add_argument_group(title='evaluation options')
    # basic evaluation settings
    eval_group.add_argument('--eval', type=bool, default=False, help='whether to run evaluation')
    # 'agora',, 'mpiinf' ,'pw3d', 'jta','h36m','pw3d','pw3d_pc','oh','h36m' # 'mupots','oh','h36m','mpiinf_test','oh',
    eval_group.add_argument('--eval_datasets', type=str, default='', help='whether to run evaluation')  # pw3d
    eval_group.add_argument('--val_batch_size', default=64, help='valiation batch_size', type=int)
    eval_group.add_argument('--only_for_have_people_crops',  default=False, help='only predict the crops which has people (by alphapose determine)', type=bool)
    
    eval_group.add_argument('--test_interval', default=2000, help='interval iteration between validation',
                            type=int)  # 2000
    eval_group.add_argument('--fast_eval_iter', type=int, default=-1,
                            help='whether to run validation on a few iterations, like 200.')
    eval_group.add_argument('--top_n_error_vis', type=int, default=6,
                            help='visulize the top n results during validation')
    eval_group.add_argument('--eval_2dpose', type=bool, default=False)
    eval_group.add_argument('--calc_pck', type=bool, default=False, help='whether calculate PCK during evaluation')
    eval_group.add_argument('--PCK_thresh', type=int, default=150, help='training epochs')
    eval_group.add_argument('--calc_PVE_error', type=bool, default=False)

    maps_group = parser.add_argument_group(title='Maps options')
    maps_group.add_argument('--centermap_size', type=int, default=64, help='the size of center map')
    maps_group.add_argument('--centermap_conf_thresh', type=float, default=0.25,
                            help='the threshold of the centermap confidence for the valid subject')
    maps_group.add_argument('--collision_aware_centermap', type=bool, default=False,
                            help='whether to use collision_aware_centermap')
    maps_group.add_argument('--collision_factor', type=float, default=0.2,
                            help='whether to use collision_aware_centermap')
    maps_group.add_argument('--center_def_kp', type=bool, default=True, help='center definition: keypoints or bbox')

    distributed_train_group = parser.add_argument_group(title='options for distributed training')
    distributed_train_group.add_argument('--local_rank', type=int, default=0,
                                         help='local rank for distributed training')
    distributed_train_group.add_argument('--distributed_training', type=bool, default=False,
                                         help='wether train model in distributed mode')

    distillation_group = parser.add_argument_group(title='options for distillation')
    distillation_group.add_argument('--distillation_learning', type=bool, default=False)
    distillation_group.add_argument('--teacher_model_path', type=str,
                                    default='/export/home/suny/CenterMesh/trained_models/3dpw_88_57.8.pkl')

    log_group = parser.add_argument_group(title='log options')
    # basic log settings
    log_group.add_argument('--print_freq', type=int, default=30, help='training epochs')
    log_group.add_argument('--model_path', type=str, default='', help='trained model path')
    log_group.add_argument('--log-path', type=str, default=os.path.join(root_dir, 'log/'), help='Path to save log file')

    hm_ae_group = parser.add_argument_group(title='learning 2D pose/associate embeddings options')
    hm_ae_group.add_argument('--learn_2dpose', type=bool, default=False)
    hm_ae_group.add_argument('--learn_AE', type=bool, default=False)
    hm_ae_group.add_argument('--learn_kp2doffset', type=bool, default=False)

    augmentation_group = parser.add_argument_group(title='augmentation options')
    # augmentation settings
    augmentation_group.add_argument('--shuffle_crop_mode', type=bool, default=False,
                                    help='whether to shuffle the data loading mode between crop / uncrop for indoor 3D pose dataset only')
    augmentation_group.add_argument('--shuffle_crop_ratio_3d', default=0.9, type=float,
                                    help='the probability of changing the data loading mode from uncrop multi_person to crop single person')
    augmentation_group.add_argument('--shuffle_crop_ratio_2d', default=0.1, type=float,
                                    help='the probability of changing the data loading mode from uncrop multi_person to crop single person')
    augmentation_group.add_argument('--Synthetic_occlusion_ratio', default=0, type=float,
                                    help='whether to use use Synthetic occlusion')
    augmentation_group.add_argument('--color_jittering_ratio', default=0.2, type=float,
                                    help='whether to use use color jittering')
    augmentation_group.add_argument('--rotate_prob', default=0.2, type=float,
                                    help='whether to use rotation augmentation')

    dataset_group = parser.add_argument_group(title='datasets options')
    # dataset setting:
    dataset_group.add_argument('--dataset_rootdir', type=str, default=os.path.join(project_dir, '..', 'data', 'train_data'),
                               help='root dir of all datasets')
    dataset_group.add_argument('--dataset', type=str, default='h36m,mpii,coco,aich,up,ochuman,lsp,movi',
                               help='which datasets are used')
    dataset_group.add_argument('--voc_dir', type=str, default=os.path.join(root_dir, 'dataset/VOCdevkit/VOC2012/'),
                               help='VOC dataset path')
    dataset_group.add_argument('--max_person', default=64, type=int, help='max person number of each image')
    dataset_group.add_argument('--homogenize_pose_space', type=bool, default=False,
                               help='whether to homogenize the pose space of 3D datasets')
    dataset_group.add_argument('--use_eft', type=bool, default=True, help='wether use eft annotations for training')

    smpl_group = parser.add_argument_group(title='SMPL options')
    # smpl info
    # smpl_group.add_argument('--smpl-mean-param-path',type = str,default = os.path.join(model_dir,'parameters','neutral_smpl_mean_params.h5'),
    #    help = 'the path for mean smpl param value')
    # smpl_group.add_argument('--smpl-model',type = str,default = os.path.join(model_dir,'parameters','neutral_smpl_with_cocoplus_reg.txt'),
    #    help = 'smpl model path')
    smpl_group.add_argument('--smpl_mesh_root_align', type=bool, default=True)
    mode_group.add_argument('--Rot_type', type=str, default='6D', help='rotation representation type: angular, 6D')
    mode_group.add_argument('--rot_dim', type=int, default=6, help='rotation representation type: 3D angular, 6D')
    mode_group.add_argument('--hvip_dim', type=int, default=1, help='the dimention of hvip param')
    smpl_group.add_argument('--trans_dim', type=int, default=3, help='the dimention of trans param')
    smpl_group.add_argument('--beta_dim', type=int, default=10, help='the dimention of SMPL shape param, beta')
    smpl_group.add_argument('--smpl_joint_num', type=int, default=22, help='joint number of SMPL model we estimate')
    smpl_group.add_argument('--smpl_model_path', type=str, default=os.path.join(model_dir, 'parameters', 'smpl'),
                            help='smpl model path')
    smpl_group.add_argument('--smpl_J_reg_h37m_path', type=str,
                            default=os.path.join(model_dir, 'parameters', 'smpl', 'J_regressor_h36m.npy'),
                            help='SMPL regressor for 17 joints from H36M datasets')
    smpl_group.add_argument('--smpl_J_reg_extra_path', type=str,
                            default=os.path.join(model_dir, 'parameters', 'smpl', 'J_regressor_extra.npy'),
                            help='SMPL regressor for 9 extra joints from different datasets')

    smpl_group.add_argument('--smpl_uvmap', type=str, default=os.path.join(model_dir, 'parameters', 'smpl', 'smpl_vt_ft.npz'),
                            help='smpl UV Map coordinates for each vertice')
    smpl_group.add_argument('--wardrobe', type=str, default=os.path.join(model_dir, 'wardrobe'),
                            help='path of smpl UV textures')
    smpl_group.add_argument('--cloth', type=str, default='f1',
                            help='pick up cloth from the wardrobe or simplely use a single color')


    debug_group = parser.add_argument_group(title='Debug options')
    debug_group.add_argument('--track_memory_usage', type=bool, default=False)

    parsed_args = parser.parse_args(args=input_args)
    parsed_args.adjust_lr_epoch = []
    parsed_args.kernel_sizes = [5]
    with open(parsed_args.configs_yml) as file:
        configs_update = yaml.full_load(file)
    for key, value in configs_update['ARGS'].items():
        if sum(['--{}'.format(key) in input_arg for input_arg in input_args]) == 0:
            if isinstance(value, str):
                exec("parsed_args.{} = '{}'".format(key, value))
            else:
                exec("parsed_args.{} = {}".format(key, value))
    if 'loss_weight' in configs_update:
        for key, value in configs_update['loss_weight'].items():
            exec("parsed_args.{}_weight = {}".format(key, value))
    if 'sample_prob' in configs_update:
        parsed_args.sample_prob_dict = configs_update['sample_prob']

    parsed_args.tab = '{}_cm{}_{}'.format(parsed_args.backbone,
                                          parsed_args.centermap_size,
                                          parsed_args.tab,
                                          parsed_args.dataset)

    return parsed_args


class ConfigContext(object):
    """
    Class to manage the active current configuration, creates temporary `yaml`
    file containing the configuration currently being used so it can be
    accessed anywhere.
    """
    yaml_filename = yaml_timestamp
    parsed_args = parse_args(sys.argv[1:])

    def __init__(self, parsed_args=None):
        if parsed_args is not None:
            self.parsed_args = parsed_args

    def __enter__(self):
        # if a yaml is left over here, remove it
        self.clean()
        # store all the parsed_args in a yaml file
        with open(self.yaml_filename, 'w') as f:
            d = self.parsed_args.__dict__
            yaml.dump(d, f)

    def __forceyaml__(self, filepath):
        # if a yaml is left over here, remove it
        self.yaml_filename = filepath
        self.clean()
        # store all the parsed_args in a yaml file
        with open(self.yaml_filename, 'w') as f:
            d = self.parsed_args.__dict__
            yaml.dump(d, f)
            print("----------------------------------------------")
            print("__forceyaml__ DUMPING YAML ")
            print("self.yaml_filename", self.yaml_filename)
            print("----------------------------------------------")

    def clean(self):
        if os.path.exists(self.yaml_filename):
            os.remove(self.yaml_filename)

    def __exit__(self, exception_type, exception_value, traceback):
        # delete the yaml file
        self.clean()


# def args():
#     return ConfigContext.parsed_args

def _get_args():
    parsed_args = parse_args(['--tab', 'v1'])
    if os.path.exists(ConfigContext.yaml_filename):
        with open(ConfigContext.yaml_filename, 'r') as f:
            argsdict = yaml.load(f, Loader=yaml.FullLoader)
    else:
        # This will write a new Yaml if the yaml doesn't exist.
        # configcontext.__forceyaml__(configcontext.yaml_filename)
        with open(ConfigContext.yaml_filename, 'w') as f:
            d = ConfigContext.parsed_args.__dict__
            yaml.dump(d, f)
        with open(ConfigContext.yaml_filename, 'r') as f:
            argsdict = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in argsdict.items():
        parsed_args.__dict__[k] = v
    return parsed_args


ARGS = _get_args()
print('load ARGS done.')


def args():
    return ARGS
