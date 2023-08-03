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

sys.path.append(os.path.abspath(__file__).replace('dataset/single_scene_image.py', ''))
from dataset.image_base import *
import config
from config import args
import constants


class SINGLE_SCENE_IMAGE(Dataset):
    def __init__(self, image_root='', ground_cam_path='',  have_people_image_json='', **kwargs):
        super(SINGLE_SCENE_IMAGE, self).__init__()
        self.image_root=image_root
        self.ground_cam_path=ground_cam_path
        self.have_people_image_json=have_people_image_json
        self.collect_file_paths()
        self.load_ground_cam()


    def collect_file_paths(self):
        self.file_paths=[]
        if os.path.exists(self.have_people_image_json):
            image_list=self.load_json(self.have_people_image_json)
        else:
            # cur_file_paths=os.listdir(self.image_folder)
            image_list=os.listdir(self.image_root)
        for patch_name in image_list:
            self.file_paths.append(os.path.join(self.image_root, patch_name))
    



    def load_ground_cam(self):
        self.ground=np.load(os.path.join(self.ground_cam_path,  'ground.npy'))
        self.ground=self.ground / np.linalg.norm(self.ground)

        self.camK=np.load(os.path.join(self.ground_cam_path, 'cam_para.npy'))
        self.scene_shape=np.load(os.path.join(self.ground_cam_path, 'scene_shape.npy'))

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

    # def __getitem__(self, index):
    #     try:
    #         return self.get_item_single_frame(index)
    #     except Exception as error:
    #         print(error)
    #         index = np.random.randint(len(self))
    #         return self.get_item_single_frame(index)

    def __getitem__(self, index):
        return self.get_item_single_frame(index)
     

    def load_pkl(self,name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def load_json(self, name):
        with open(name, 'r') as rf:
            return json.load(rf)


    def img_preprocess(self, img_path, input_size=512, ds='single_scene_image'):
        image=cv2.imread(img_path)
        image = image[:, :, ::-1]
        input_data={}
        
        

        name = os.path.basename(img_path)
        input_data.update({'imgpath': img_path, 'name': name})
        

        infos=name.split('_')
        h1, h2 = int(infos[-4]), int(infos[-2])
        w1, w2 = int(infos[-3]), int(infos[-1][:-4])
        cur_size = h2 - h1


        patch_top, patch_left = h1, w1
        patch_leftTop = np.array([patch_left, patch_top]).astype(int)
        
        image_org, offsets = process_image(image, None)
        data_scale=cur_size / 512
        offsets.append(data_scale)
        offsets.append(self.scene_shape[0]) # scene_w
        offsets.append(self.scene_shape[1]) # scene_h
        
        image = torch.from_numpy(cv2.resize(image_org.copy(), (input_size, input_size), interpolation=cv2.INTER_CUBIC))
        offsets = torch.from_numpy(np.array(offsets)).double()

        input_data['image']=image.float()
        input_data['offsets']=offsets
        input_data['data_set']=ds
        
        input_data['ground'] = torch.from_numpy(self.ground).double()# np.array(input_params['ground'])
        input_data['camK'] = torch.from_numpy(self.camK).double()
        input_data['dist'] = torch.from_numpy(np.zeros(5)).double()
        input_data['patch_leftTop'] = torch.from_numpy(patch_leftTop).double()
        return input_data

