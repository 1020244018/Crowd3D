import glob
import json

import numpy as np
import random
import cv2
import torch
import shutil
import time
import copy
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
import sys, os, pickle

sys.path.append(os.path.abspath(__file__).replace('dataset/eval_largecrowd.py', ''))
from dataset.image_base import *
import config
from config import args
import constants



class EVAL_LargeCrowd(Dataset):
    def __init__(self, data_root='', ground_cam_root='', scene_type='',  **kwargs):
        super(EVAL_LargeCrowd, self).__init__()
        self.eval_scene=scene_type
        self.each_image_folder='part_images_scale'
        self.data_root=data_root
        self.ground_cam_root=ground_cam_root
        self.collect_file_paths(data_root)
        self.load_ground_cam()


    def collect_file_paths(self, data_root):
        self.file_paths=[]
        scene_type=self.eval_scene
        frame_list=os.listdir(os.path.join(data_root, scene_type))
        for frame_name in frame_list:
            have_people_image_json=os.path.join(data_root, scene_type, frame_name, 'have_people_image_name.json')
            if os.path.exists(have_people_image_json) and args().only_for_have_people_crops:
                cur_file_paths=self.load_json(have_people_image_json)
            else:
                # cur_file_paths=os.listdir(self.image_folder)
                cur_file_paths=os.listdir(os.path.join(data_root, scene_type, frame_name, self.each_image_folder))
            for patch_name in cur_file_paths:
                self.file_paths.append(os.path.join(data_root, scene_type, frame_name, self.each_image_folder, patch_name))

    def load_ground_cam(self):
        self.ground={}
        self.cam={}
        self.scene_shape={}
        for scene_type in [self.eval_scene]:
            self.ground[scene_type]=np.load(os.path.join(self.ground_cam_root, scene_type, 'ground.npy'))
            self.cam[scene_type]=np.load(os.path.join(self.ground_cam_root, scene_type, 'cam_para.npy'))
            self.scene_shape[scene_type]=np.load(os.path.join(self.ground_cam_root, scene_type, 'scene_shape.npy'))

    def get_image_info(self, index):
        return self.file_paths[index]

    def resample(self):
        return random.randint(0, len(self))

    def get_item_single_frame(self, index):

        img_path = self.file_paths[index]
        input_data = self.img_preprocess(img_path, input_size=args().input_size)

        return input_data

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        try:
            return self.get_item_single_frame(index)
        except Exception as error:
            print(error)
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)

    def load_pkl(self,name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def load_json(self, name):
        with open(name, 'r') as rf:
            return json.load(rf)


    def img_preprocess(self, img_path, input_size=512, ds='eval_largecrowd'):
        image=cv2.imread(img_path)
        image = image[:, :, ::-1]
        input_data={}

        name = os.path.basename(img_path)
        input_data.update({'imgpath': img_path, 'name': name})

        infos=name.split('_')
        h1, h2 = int(infos[-4]), int(infos[-2])
        w1, w2 = int(infos[-3]), int(infos[-1][:-4])
        cur_size = h2 - h1
        scene_type=infos[0]+'_'+infos[1]
        camK=self.cam[scene_type].astype(np.float)
        ground=self.ground[scene_type]
        scene_shape=self.scene_shape[scene_type]

        patch_top, patch_left = h1, w1
        patch_leftTop = np.array([patch_left, patch_top]).astype(int)
        


        image_org,  offsets = process_image(image)
        data_scale=cur_size / 512
        offsets.append(data_scale)
        offsets.append(scene_shape[0]) # scene_w
        offsets.append(scene_shape[1]) # scene_h
        
        image = torch.from_numpy(image_org.copy())
        offsets = torch.from_numpy(np.array(offsets)).double()

        input_data['image']=image.float()
        input_data['offsets']=offsets
        input_data['data_set']=ds
        
        input_data['ground'] = torch.from_numpy(ground).double()# np.array(input_params['ground'])
        input_data['camK'] = torch.from_numpy(camK).double()
        input_data['dist'] = torch.from_numpy(np.zeros(5)).double()
        input_data['patch_leftTop'] = torch.from_numpy(patch_leftTop).double()
    

        return input_data

