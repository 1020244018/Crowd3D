import sys, os
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch
import shutil
import time
import pickle
import copy
import joblib
import logging
import scipy.io as scio
import quaternion

from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from models.smpl_regressor import SMPLR
from utils import Synthetic_occlusion, process_image, calc_aabb, flip_kps, rot_imgplane, pose_processing
from maps_utils import HeatmapGenerator, JointsGenerator, CenterMap
from config import args
import config
import constants
from utils.center_utils import denormalize_center
from maps_utils.centermap import _calc_radius_


class Image_base(Dataset):
    def __init__(self, train_flag=True, regress_smpl=False):
        super(Image_base, self).__init__()
        self.data_folder = args().dataset_rootdir
        self.scale_range = [1.2, 1.7]
        self.half_prob = 0.12
        self.noise = 0.2
        self.vis_thresh = 0.03
        self.channels_mix = False
        self.ID_num = 0
        self.min_vis_pts = 2

        self.max_person = args().max_person
        self.multi_mode = args().multi_person
        self.use_eft = args().use_eft
        self.regress_smpl = regress_smpl

        self.homogenize_pose_space = False
        self.max_diff=0
        if train_flag:
            self.homogenize_pose_space = args().homogenize_pose_space
            if args().Synthetic_occlusion_ratio > 0:
                self.synthetic_occlusion = Synthetic_occlusion(args().voc_dir)
            if args().color_jittering_ratio > 0:
                self.color_jittering = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                                          hue=0)

        # only shuffle some 3D pose datasets, such as h36m and mpi-inf-3Dhp
        self.shuffle_mode = args().shuffle_crop_mode
        self.shuffle_ratio = args().shuffle_crop_ratio_2d
        self.train_flag = train_flag

        self.input_shape = [args().input_size, args().input_size]
        self.vis_size = args().input_size
        self.labels, self.images, self.file_paths = [], [], []

        self.root_inds = [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]
        self.neck_idx, self.pelvis_idx = 1, 8
        # self.torso_ids = [constants.SMPL_ALL_54[part] for part in
        #                   ['Neck', 'Neck_LSP', 'R_Shoulder', 'L_Shoulder', 'Pelvis', 'R_Hip', 'L_Hip']]
        # self.torso_ids = [constants.SMPL_ALL_54[part] for part in
        #                   ['Neck', 'R_Shoulder', 'L_Shoulder', 'Pelvis', 'R_Hip', 'L_Hip']]
        self.torso_ids = [constants.SMPL_ALL_54[part] for part in
                          ['R_Shoulder', 'L_Shoulder', 'R_Hip', 'L_Hip']]
        
        self.phase = 'train' if self.train_flag else 'test'
        self.default_valid_mask_3d = np.array([False for _ in range(4)])
        self.heatmap_res = 128
        self.joint_number = len(list(constants.SMPL_ALL_54.keys()))

        if args().learn_2dpose:
            self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.COCO_17)
            self.heatmap_generator = HeatmapGenerator(self.heatmap_res, len(self.heatmap_mapper))
        if args().learn_AE:
            self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.COCO_17)
            self.joint_generator = JointsGenerator(self.max_person, len(self.heatmap_mapper), 128, True)
        self.CM = CenterMap()

    def get_content_for_test(self, index):
        info = self.get_image_info(index)
        
        img_info = process_image(info['image'], info['kp2ds'], full_bboxs=info['bbox'], foot2ds=info['foot2ds'], ankle2ds=info['ankle2ds'],
                                 augments=None,
                                 is_pose2d=info['vmask_2d'][:, 0], multiperson=True)
        image, image_wbg, full_kps, offsets, full_bboxs, foot2ds = img_info
        image = self.prepare_image2(image)
        return info, image

    
    def get_item_single_frame(self, index):
        # valid annotation flags for 
        # 0: 2D pose/bounding box(True/False), # detecting all person/front-view person(True/False)
        # 1: 3D pose, 2: subject id, 3: smpl root rot, 4: smpl pose param, 5: smpl shape param
        valid_masks = np.zeros((self.max_person, 10), dtype=np.bool)
        info = self.get_image_info(index)
        scale, rot, flip, color_jitter, syn_occlusion = self._calc_csrfe()
        # print('scale, rot, flip, color_jitter, syn_occlusion', scale, rot, flip, color_jitter, syn_occlusion )
        flip = False
        mp_mode = True  # self._check_mp_mode_()
        h, w = info['img_size']
        if 'real_tcs' not in info:
            info['real_tcs']=None
        img_info = process_image(info['image'], info['kp2ds'], full_bboxs=info['bbox'], foot2ds=info['hvip2ds'], ankle2ds=info['ankle2ds'],
                                 augments=(scale, rot, flip), is_pose2d=info['vmask_2d'][:, 0], 
                                 multiperson=mp_mode, data_scale=info['data_scale'], real_cps=info['real_tcs'], scene_shape=info['scene_shape'])

        image, image_wbg, full_kps, offsets, full_bboxs, hvip2ds, ankle2ds, real_tcs = img_info

        centermap, person_centers, full_kp2ds, used_person_inds, valid_masks[:,
                                                                 0], bboxes_hw_norm, heatmap, AE_joints, full_bbox2ds, full_hvip2ds, full_ankle2ds,  valid_masks[
                                                                                                                                     :,
                                                                                                                                     6], valid_masks[:, 8], full_real_tcs2ds = self.process_kp2ds_bboxes(
            full_kps, full_bboxs=full_bboxs, foot2ds=hvip2ds, ankle2ds=ankle2ds, img_shape=image.shape, is_pose2d=info['vmask_2d'][:, 0],
            foot2d_flag=info['vmask_foot'][:, 0], ankle_flag=info['vmask_foot'][:, 2], real_cps=real_tcs)
        all_person_detected_mask = info['vmask_2d'][0, 2]
        subject_ids, valid_masks[:, 2] = self.process_suject_ids(info['track_ids'], used_person_inds,
                                                                 valid_mask_ids=info['vmask_2d'][:, 1])
        image, dst_image, org_image = self.prepare_image(image, image_wbg, augments=(color_jitter, syn_occlusion))


        # print('prepare_image image.shape', image.shape, 'dst_image.shape',dst_image.shape, 'org_image.shape', org_image.shape)
        # valid mask of 3D pose, smpl root rot, smpl pose param, smpl shape param, global translation
        kp3d, valid_masks[:, 1] = self.process_kp3ds(info['kp3ds'], used_person_inds, \
                                                     augments=(rot, flip), valid_mask_kp3ds=info['vmask_3d'][:, 0])

        all_trans, valid_masks[:, 7] = self.process_trans(info['trans'], used_person_inds, \
                                                          augments=None, valid_mask_trans=info['vmask_foot'][:, 1])

        all_root_cam, valid_masks[:, 9] = self.process_trans(info['root_cam'], used_person_inds, \
                                                          augments=None, valid_mask_trans=info['vmask_foot'][:, 3])

        params, valid_masks[:, 3:6] = self.process_smpl_params(info['params'], used_person_inds, \
                                                               augments=(rot, flip),
                                                               valid_mask_smpl=info['vmask_3d'][:, 1:4])

        camk_new = info['camK']
        if flip:
            camk_new[0, 2] = w - camk_new[0, 2]

        full_ankle_means=np.mean(full_ankle2ds, axis=1)
        delta2d=full_ankle_means-full_hvip2ds # 64,2
        delta2d[~valid_masks[:,6]]=-2
 


        ground=info['ground'] # 4,
        if ground[1]>0:
            ground=-ground
        ground=ground / np.linalg.norm(ground)

        input_data = {
            'image': torch.from_numpy(dst_image).float(),
            'image_org': torch.from_numpy(org_image),
            'full_kp2d': torch.from_numpy(full_kp2ds).float(),
            'person_centers': torch.from_numpy(person_centers).float(),
            'subject_ids': torch.from_numpy(subject_ids).long(),
            'centermap': centermap.float(),
            'kp_3d': torch.from_numpy(kp3d).float(),
            'params': torch.from_numpy(params).float(),
            'valid_masks': torch.from_numpy(valid_masks).bool(),
            'offsets': torch.from_numpy(offsets).float(),
            'rot_flip': torch.Tensor([rot, flip]).float(),
            'all_person_detected_mask': torch.Tensor([all_person_detected_mask]).bool(),
            'imgpath': info['imgpath'],
            'data_set': info['ds'],
            'ground': torch.from_numpy(ground).float(),
            'camK': torch.from_numpy(camk_new).double(),  # need to keep accuracy.
            'dist': torch.from_numpy(info['dist']).float(),
            'patch_leftTop': info['patch_leftTop'],
            'hvip2ds': torch.from_numpy(full_hvip2ds).float(),
            'trans3d': torch.from_numpy(all_trans).float(),
            'delta2d': torch.from_numpy(delta2d).float(),
            'root_cam': torch.from_numpy(all_root_cam).float(),
            'ankle2ds': torch.from_numpy(full_ankle2ds).float(),
            'real_tcs':torch.from_numpy(full_real_tcs2ds).float()
            }

        if args().learn_2dpose:
            input_data.update({'heatmap': torch.from_numpy(heatmap).float()})
        if args().learn_AE:
            input_data.update({'AE_joints': torch.from_numpy(AE_joints).long()})

        return input_data

    def process_kps(self, kps, img_size, set_minus=True):
        kps = kps.astype(np.float32)
        kps[:, 0] = kps[:, 0] / float(img_size[1])
        kps[:, 1] = kps[:, 1] / float(img_size[0])
        kps[:, :2] = 2.0 * kps[:, :2] - 1.0
        if set_minus:
            if kps.shape[1] > 2:
                kps[kps[:, 2] <= self.vis_thresh] = -2.
            kps = kps[:, :2]
            for inds, kp in enumerate(kps):
                if not _check_upper_bound_lower_bound_(kp, ub=1.5, lb=-1.5):  # for truncation
                    kps[inds] = -2.
        return kps

    def process_cps(self, cps, img_size):
        kps=cps
        kps = kps.astype(np.float32)
        kps[:, 0] = kps[:, 0] / float(img_size[1])
        kps[:, 1] = kps[:, 1] / float(img_size[0])
        kps[:, :2] = 2.0 * kps[:, :2] - 1.0

        return kps

    def process_bbox(self, bbox, img_size):
        bbox = np.array(bbox).astype(np.float)
        bbox[0] = bbox[0] / float(img_size[1])
        bbox[1] = bbox[1] / float(img_size[0])
        bbox[2] = bbox[2] / float(img_size[1])
        bbox[3] = bbox[3] / float(img_size[0])
        bbox = bbox * 2.0 - 1.
        return bbox

    def process_foot2d(self, foot2d, img_size):
        foot2d = np.array(foot2d).astype(np.float)
        foot2d[0] = foot2d[0] / float(img_size[1])
        foot2d[1] = foot2d[1] / float(img_size[0])
        foot2d = foot2d * 2.0 - 1.
        return foot2d

    def map_kps(self, joint_org, maps=None):
        kps = joint_org[maps].copy()
        kps[maps == -1] = -2.
        return kps

    def parse_multiperson_kp2ds(self, full_kps, full_bbox=None, foot2ds=None, ankle2ds=None, foot2d_flag=None, ankle_flag=None, real_cps=None):
        bboxes_normalized = _calc_bbox_normed(full_kps)  # [leftTop, rightBottom]
        if args().learn_2dpose or args().learn_AE:
            heatmap, AE_joints = self.generate_heatmap_AEmap(full_kps)
        else:
            heatmap, AE_joints = np.zeros((17, 128, 128)), np.zeros((self.max_person, 17, 2))
        person_centers, full_kp2ds, bboxes_hw_norm, used_person_inds = [], [], [], []

        if full_bbox is not None:
            full_bbox2ds = []
        else:
            full_bbox2ds = None

        if foot2ds is not None:
            foot2ds_2ds = []
        else:
            foot2ds_2ds = None
        
        if ankle2ds is not None:
            ankle2ds_2ds=[]
        else:
            ankle2ds_2ds=None

        if real_cps is not None:
            real_cps_2d_keep=[]
        else:
            real_cps_2d_keep=None
        
        foot2d_flag_2ds = []
        ankle_flag_2ds=[]
        
        for inds in range(min(len(full_kps), self.max_person)):
            center = self._calc_center_(full_kps[inds])
            if center is None or len(center) == 0:
                continue
            person_centers.append(center)
            if real_cps is not None:
                real_cps_2d_keep.append(real_cps[inds][0])
            full_kp2ds.append(full_kps[inds])
            foot2d_flag_2ds.append(foot2d_flag[inds])
            ankle_flag_2ds.append(ankle_flag[inds])
            if full_bbox is not None:
                full_bbox2ds.append(full_bbox[inds])
            if foot2ds is not None:
                foot2ds_2ds.append(foot2ds[inds])
            if ankle2ds is not None:
                ankle2ds_2ds.append(ankle2ds[inds])
            bboxes_hw_norm.append((bboxes_normalized[inds][1] - bboxes_normalized[inds][0])[::-1])
            used_person_inds.append(inds)

        person_centers, full_kp2ds = np.array(person_centers), np.array(full_kp2ds)
        foot2d_flag_2ds = np.array(foot2d_flag_2ds)
        ankle_flag_2ds = np.array(ankle_flag_2ds)

        if full_bbox is not None:
            full_bbox2ds = np.array(full_bbox2ds)
        if foot2ds is not None:
            foot2ds_2ds = np.array(foot2ds_2ds)
        if ankle2ds is not None:
            ankle2ds_2ds = np.array(ankle2ds_2ds)
        if real_cps_2d_keep is not None:
            real_cps_2d_keep=np.array(real_cps_2d_keep)
        occluded_by_who = detect_occluded_person(person_centers,
                                                 full_kp2ds) if args().collision_aware_centermap else None
        
        return person_centers, full_kp2ds, used_person_inds, bboxes_hw_norm, occluded_by_who, heatmap, AE_joints, full_bbox2ds, foot2ds_2ds, ankle2ds_2ds, foot2d_flag_2ds, ankle_flag_2ds, real_cps_2d_keep

    def parse_bboxes(self, full_kps, hw_ratio_thresh=0.8):
        person_centers, bboxes_hw_norm, used_person_inds = [], [], []
        fbox, vbox = full_kps[:, :2], full_kps[:, 2:]
        fwh, vwh = fbox[:, 1] - fbox[:, 0], vbox[:, 1] - vbox[:, 0]
        fhw_ratios, vhw_ratios = fwh[:, 1] / (fwh[:, 0] + 1e-4), vwh[:, 1] / (vwh[:, 0] + 1e-4)
        fb_centers = np.stack([0.5 * fbox[:, 0, 0] + 0.5 * fbox[:, 1, 0], 0.7 * fbox[:, 0, 1] + 0.3 * fbox[:, 1, 1]], 1)
        vb_centers = np.stack([0.5 * vbox[:, 0, 0] + 0.5 * vbox[:, 1, 0], 0.56 * vbox[:, 0, 1] + 0.44 * vbox[:, 1, 1]],
                              1)
        for inds, (fhw_ratio, vhw_ratio) in enumerate(zip(fhw_ratios, vhw_ratios)):
            if len(used_person_inds) >= self.max_person:
                continue
            if vhw_ratio > 1. and _check_upper_bound_lower_bound_(fb_centers[inds], ub=1, lb=-1):
                person_centers.append(fb_centers[inds])
                bboxes_hw_norm.append(fwh[inds][::-1])
                used_person_inds.append(inds)
            elif _check_upper_bound_lower_bound_(vb_centers[inds], ub=1, lb=-1):
                person_centers.append(vb_centers[inds])
                bboxes_hw_norm.append(vwh[inds][::-1])
                used_person_inds.append(inds)

        return person_centers, bboxes_hw_norm, used_person_inds

    def process_kp2ds_bboxes(self, full_kps, full_bboxs=None, foot2ds=None, ankle2ds=None, img_shape=None, is_pose2d=None,
                             foot2d_flag=None, ankle_flag=None, real_cps=None):
        person_centers = np.ones((self.max_person, 2)) * -2.
        full_bbox2ds = None #np.ones((self.max_person, 4)) * -2.
        full_foot2ds = np.ones((self.max_person, 2)) * -2.
        full_ankle2ds = np.ones((self.max_person, 2, 2))* -2
        full_kp2ds = np.ones((self.max_person, self.joint_number, 2)) * -2.
        valid_mask_kp2ds = np.zeros(self.max_person, dtype=np.bool)
        valid_mask_foot2d = np.zeros(self.max_person, dtype=np.bool)
        valid_mask_ankle = np.zeros(self.max_person, dtype=np.bool)
        used_person_inds, bboxes_hw_norm, occluded_by_who = [], [], None
        full_real_cps2ds=np.ones((self.max_person, 2)) * -2.

        if is_pose2d.sum() > 0:
            full_kp2d = [self.process_kps(full_kps[ind], img_shape) for ind in np.where(is_pose2d)[0]]
            full_bbox2d = None
            foot2ds_2d = None
            ankle2ds_2d = None
            real_cps_2d=None
            if full_bboxs is not None:
                full_bbox2d = [self.process_bbox(full_bboxs[ind], img_shape) for ind in np.where(is_pose2d)[0]]
            if foot2ds is not None:
                foot2ds_2d = [self.process_foot2d(foot2ds[ind], img_shape) for ind in np.where(is_pose2d)[0]]
            if ankle2ds is not None:
                ankle2ds_2d = [self.process_kps(ankle2ds[ind], img_shape, set_minus=False) for ind in np.where(is_pose2d)[0]]
            if real_cps is not None:
                real_cps_2d=[self.process_cps(real_cps[ind], img_shape) for ind in np.where(is_pose2d)[0]]
            person_centers_kp2d, full_kp2ds_kp2d, mask_kp2d, bboxes_hw_norm_kp2d, occluded_by_who, heatmap, AE_joints, full_bbox2d_bbox2d, foot2ds_2ds_bbox2d, ankle2ds_2ds_bbox2d, foot2d_flag_bbox2d, ankle_flag_bbox2d, real_cps_kp2d = self.parse_multiperson_kp2ds(
                full_kp2d, full_bbox=full_bbox2d, foot2ds=foot2ds_2d, ankle2ds=ankle2ds_2d, foot2d_flag=foot2d_flag, ankle_flag=ankle_flag, real_cps=real_cps_2d)

            mask_kp2d = np.where(is_pose2d)[0][np.array(mask_kp2d, dtype=np.int32)].tolist()

            if len(mask_kp2d) > 0:
                person_centers[:len(mask_kp2d)], full_kp2ds[:len(mask_kp2d)], valid_mask_kp2ds[:len(
                    mask_kp2d)] = person_centers_kp2d, full_kp2ds_kp2d, True
                valid_mask_foot2d[:len(mask_kp2d)] = foot2d_flag_bbox2d
                valid_mask_ankle[:len(mask_kp2d)] = ankle_flag_bbox2d
                if full_bboxs is not None:
                    full_bbox2ds[:len(mask_kp2d)] = full_bbox2d_bbox2d
                if foot2ds is not None:
                    full_foot2ds[:len(mask_kp2d)] = foot2ds_2ds_bbox2d
                if ankle2ds is not None:
                    full_ankle2ds[:len(mask_kp2d)] = ankle2ds_2ds_bbox2d
                if real_cps is not None:
                    full_real_cps2ds[:len(mask_kp2d)]=real_cps_kp2d

                used_person_inds += mask_kp2d
                bboxes_hw_norm += bboxes_hw_norm_kp2d
        if (~is_pose2d).sum() > 0:
            full_bboxes = np.array(
                [self.process_kps(full_kps[ind], img_shape, set_minus=False) for ind in np.where(~is_pose2d)[0]])
            person_centers_bbox, bboxes_hw_norm_bbox, mask_bbox = self.parse_bboxes(full_bboxes)
            mask_bbox = np.where(~is_pose2d)[0][np.array(mask_bbox, dtype=np.int32)].tolist()
            left_num = max(0, min(self.max_person - len(used_person_inds), len(mask_bbox)))
            if left_num != len(mask_bbox):
                person_centers_bbox, bboxes_hw_norm_bbox, mask_bbox = person_centers_bbox[
                                                                      :left_num], bboxes_hw_norm_bbox[
                                                                                  :left_num], mask_bbox[:left_num]
            if len(mask_bbox) > 0:
                person_centers[len(used_person_inds):len(used_person_inds) + left_num] = person_centers_bbox
                used_person_inds += mask_bbox
                bboxes_hw_norm += bboxes_hw_norm_bbox

        if is_pose2d.sum() == 0:
            heatmap, AE_joints = np.zeros((17, 128, 128)), np.zeros((self.max_person, 17, 2))
        # person_centers changed after CAR processing

        centermap = self.CM.generate_centermap(person_centers, bboxes_hw_norm=bboxes_hw_norm,
                                               occluded_by_who=occluded_by_who)
        # check max offset2d
        # max_diff=np.abs(person_centers[:len(mask_kp2d)] - real_cps_kp2d[:len(mask_kp2d)]).max()
        # if self.max_diff < max_diff:
        #     self.max_diff=max_diff
        #     print('update max_diff=',self.max_diff)

        # rectify the x, y order, from x-y to y-x
        person_centers = person_centers[:, ::-1].copy()
        

        return centermap, person_centers, full_kp2ds, used_person_inds, valid_mask_kp2ds, bboxes_hw_norm, heatmap, AE_joints, full_bbox2ds, full_foot2ds, full_ankle2ds, valid_mask_foot2d, valid_mask_ankle, full_real_cps2ds

    def process_suject_ids(self, subject_ids, used_person_inds, valid_mask_ids=False):
        person_ids, valid_id_mask = np.ones(self.max_person) * -1, np.zeros(self.max_person, dtype=np.bool)
        return person_ids, valid_id_mask

    def prepare_image(self, image, image_wbg, augments=None):
        color_jitter, syn_occlusion = augments
        image = self.aug_image(image, color_jitter, syn_occlusion)
        dst_image = cv2.resize(image, tuple(self.input_shape), interpolation=cv2.INTER_CUBIC)
        org_image = cv2.resize(image_wbg, (self.vis_size, self.vis_size), interpolation=cv2.INTER_CUBIC)
        return image, dst_image, org_image
    
    def prepare_image2(self, image):
        dst_image = cv2.resize(image, tuple(self.input_shape), interpolation=cv2.INTER_CUBIC)
        return dst_image
    
    # def prepare_image_gn(self, image):
    #     if image is None:
    #         return None
    #     image=image.astype(np.float32)
    #     dst_image = cv2.resize(image, tuple(self.input_shape), interpolation=cv2.INTER_CUBIC)
    #     return dst_image

    def process_kp3ds(self, kp3ds, used_person_inds, augments=None, valid_mask_kp3ds=None):
        rot, flip = augments
        kp3d_flag = np.zeros(self.max_person, dtype=np.bool)
        joint_num = self.joint_number if self.train_flag or kp3ds is None else kp3ds[0].shape[0]
        kp3d_processed = np.ones((self.max_person, joint_num, 3),
                                 dtype=np.float32) * -2.  # -2 serves as an invisible flag

        for inds, used_id in enumerate(used_person_inds):
            if valid_mask_kp3ds[used_id]:
                kp3d, kp3d_flag[inds] = kp3ds[used_id], valid_mask_kp3ds[used_id]
                valid_mask = self._check_kp3d_visible_parts_(kp3d)
                if self.root_inds is not None:
                    kp3d -= kp3d[self.root_inds].mean(0)[None]
                kp3d = rot_imgplane(kp3d, rot)
                if flip:
                    kp3d = flip_kps(kp3d, flipped_parts=constants.All54_flip)
                    valid_mask = valid_mask[constants.All54_flip]
                kp3d[~valid_mask] = -2.
                kp3d_processed[inds] = kp3d

        return kp3d_processed, kp3d_flag

    def process_trans(self, all_trans, used_person_inds, augments=None, valid_mask_trans=None):
        # not deal with augments
        trans_flag = np.zeros(self.max_person, dtype=np.bool)
        all_trans_processed = np.ones((self.max_person, 3), dtype=np.float32) * -2.  # -2 serves as an invisible flag
        for inds, used_id in enumerate(used_person_inds):
            if valid_mask_trans[used_id]:
                all_trans_processed[inds], trans_flag[inds] = all_trans[used_id], valid_mask_trans[used_id]

        return all_trans_processed, trans_flag

    def process_smpl_params(self, params, used_person_inds, augments=None, valid_mask_smpl=None):
        rot, flip = augments
        params_processed = np.ones((self.max_person, 76), dtype=np.float32) * -10
        smpl_flag = np.zeros((self.max_person, 3), dtype=np.bool)

        for inds, used_id in enumerate(used_person_inds):
            if valid_mask_smpl[used_id].sum() > 0:
                param, smpl_flag[inds] = params[used_id], valid_mask_smpl[used_id]
                theta, beta = param[:66], param[-10:]
                params_processed[inds] = np.concatenate([pose_processing(theta, rot, flip, smpl_flag[inds]), beta])

        return params_processed, smpl_flag

    def regress_kp3d_from_smpl(self, params, maps=None, genders=None):
        kp3ds = None
        if params is not None and self.regress_smpl:
            kp3ds = []
            for inds, param in enumerate(params):
                if param is not None:
                    pose, beta = np.concatenate([param[:-10], np.zeros(6)]), param[-10:]
                    gender = 'n' if genders is None else genders[inds]
                    outputs = self.smplr(pose, beta, gender=gender)
                    kp3d = outputs['j3d'][0].numpy()
                    if maps is not None:
                        kp3d = self.map_kps(kp3d, maps=maps)
                    kp3ds.append(kp3d)
                else:
                    kp3ds.append(None)
        return kp3ds

    def generate_heatmap_AEmap(self, full_kps):
        heatmaps, AE_joints = None, None
        if args().learn_2dpose or args().learn_AE:
            full_kps_hm = [(kps_i + 1.) / 2. * self.heatmap_res for kps_i in full_kps]
            full_kps_hm = [np.concatenate([kps_i, (kps_i[:, 0] > 0)[:, None]], -1) for kps_i in full_kps_hm]
            heatmap_kps = []
            for kps in full_kps_hm:
                heatmap_kps.append(kps[self.heatmap_mapper])

        if args().learn_2dpose:
            heatmaps = self.heatmap_generator.single_process(heatmap_kps)
        if args().learn_AE:
            AE_joints = self.joint_generator.single_process(heatmap_kps)
        return heatmaps, AE_joints

    def _check_kp3d_visible_parts_(self, kp3ds, invisible_flag=-2.):
        visible_parts_mask = (kp3ds != invisible_flag).sum(-1) == kp3ds.shape[-1]
        return visible_parts_mask

    def _check_mp_mode_(self):
        multi_person = True
        if not self.multi_mode:
            multi_person = False
        if self.multi_mode and self.train_flag:
            # lower self.shuffle_ratio, greater chance to be multiperson (uncrop)
            if self.shuffle_mode and random.random() < self.shuffle_ratio:
                multi_person = False
        return multi_person

    def parse_cluster_results(self, cluster_results_file, file_paths):
        annots = np.load(cluster_results_file, allow_pickle=True)
        cluster_results, img_names = annots['kp3ds'], annots['img_names']
        cluster_dict = {os.path.basename(img_name): cluster_id for img_name, cluster_id in
                        zip(img_names, cluster_results)}
        cluster_num = max(cluster_results) + 1
        cluster_pool = [[] for _ in range(cluster_num)]
        for inds, img_name in enumerate(file_paths):
            cluster_pool[cluster_dict[os.path.basename(img_name)]].append(inds)
        # for train use small data
        cluster_pool_new = [x for x in cluster_pool if x != []]
        return cluster_pool_new
        # return cluster_pool

    def homogenize_pose_sample(self, index):
        cluster_num = len(self.cluster_pool)
        return random.sample(self.cluster_pool[index % cluster_num], 1)[0]

    def get_image_info(self, index):
        raise NotImplementedError

    def resample(self):
        return self.__getitem__(random.randint(0, len(self)))

    def reget_info(self):
        return self.get_image_info(random.randint(0, len(self)))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.get_item_single_frame(index)
        try:
            return self.get_item_single_frame(index)
        except Exception as error:
            logging.error(error)
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)

    def aug_image(self, image, color_jitter, syn_occlusion):
        if syn_occlusion:
            image = self.synthetic_occlusion(image)
        if color_jitter:
            image = np.array(self.color_jittering(Image.fromarray(image)))

        return image

    def read_pkl(self, file_path):
        return pickle.load(open(file_path, 'rb'), encoding='iso-8859-1')

    def read_json(self, file_path):
        with open(file_path, 'r') as f:
            file = json.load(f)
        return file

    def read_npy(self, file_path):
        return np.load(file_path)

    def _calc_csrfe(self):
        scale = self.scale_range[0]
        rot, flip, color_jitter, syn_occlusion = 0, False, False, False
        if self.train_flag:
            scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            if self.channels_mix:
                pn = np.random.uniform(1 - self.noise, 1 + self.noise, 3)
            flip = True if random.random() < 0.5 else False
            color_jitter = True if random.random() < args().color_jittering_ratio else False
            syn_occlusion = True if random.random() < args().Synthetic_occlusion_ratio else False
            # rot = random.randint(-30, 30) if random.random() < args().rotate_prob else 0 # ����Ϊ������ת��Ӱ��ͬ��������ĵ���͸��ͶӰ��

        return scale, rot, flip, color_jitter, syn_occlusion

    def _calc_center_(self, kps):
        center = None
        if args().center_def_kp:
            vis = kps[self.torso_ids, 0] > -1
            if vis.sum() > 0:
                center = kps[self.torso_ids][vis].mean(0)
            elif (kps[:, 0] > -1).sum() > 4: # orign:0
                center = kps[kps[:, 0] > -1].mean(0)
        else:
            vis = kps[:, 0] > -1
            if vis.sum() > self.min_vis_pts:
                center = kps[vis].mean(0)
        return center


def detect_occluded_person(person_centers, full_kp2ds, thresh=2 * 64 / 512.):
    person_num = len(person_centers)
    # index of the person at the front of an occlusion
    occluded_by_who = np.ones(person_num) * -1
    if person_num > 1:
        for inds, (person_center, kp2d) in enumerate(zip(person_centers, full_kp2ds)):
            dist = np.sqrt(((person_centers - person_center) ** 2).sum(-1))
            if (dist > 0).sum() > 0:
                if (dist[dist > 0] < thresh).sum() > 0:
                    # Comparing the visible keypoint number to justify whether the person is occluded by the others
                    if (full_kp2ds[inds, :, 0] > 0).sum() >= (kp2d[:, 0] > 0).sum():
                        closet_idx = np.where(dist == np.min(dist[dist > 0]))[0][0]
                        if occluded_by_who[closet_idx] < 0:
                            occluded_by_who[inds] = closet_idx

    return occluded_by_who.astype(np.int)


def _calc_bbox_normed(full_kps):
    bboxes = []
    for kps_i in full_kps:
        if (kps_i[:, 0] > -2).sum() > 0:
            bboxes.append(calc_aabb(kps_i[kps_i[:, 0] > -2]))
        else:
            bboxes.append(np.zeros((2, 2)))

    return bboxes


def _check_minus2_error_(kp3ds, acceptable_list=[-2., 0.]):
    kp3ds_flatten = kp3ds[:, 1:].reshape(-1, 3)
    for kp3d in kp3ds_flatten:
        if kp3d[0] in acceptable_list and kp3d[1] in acceptable_list and kp3d[2] in acceptable_list:
            continue

        equal_kp_value = (kp3ds_flatten[:, 0] == kp3d[0]).long() + (kp3ds_flatten[:, 1] == kp3d[1]).long() + (
                kp3ds_flatten[:, 2] == kp3d[2]).long()
        equal_mask = equal_kp_value == 3
        if equal_mask.sum() > 3:
            print(torch.where((kp3ds[:, :] == kp3d).sum(-1) == 3))
            print(
                'there are incorrect process that may break the state of invisible flag -2., and make the value becomes {}'.format(
                    kp3ds_flatten[equal_mask]))
            # raise Exception


def _check_upper_bound_lower_bound_(kps, ub=1, lb=-1):
    for k in kps:
        if k >= ub or k <= lb:
            return False
    return True


def check_and_mkdir(dir):
    os.makedirs(dir, exist_ok=True)


def test_projection_depth(pred_joints, trans_xyz, depth_pred, fov_tan):
    from utils.projection import batch_persp_depth
    projected_kp2d = batch_persp_depth(pred_joints, trans_xyz, depth_pred, fov_tan)
    return projected_kp2d


def denormalize_kp2ds(mat, img_size=args().input_size):
    return (mat + 1) / 2 * img_size


def print_data_shape(data):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        elif isinstance(value, list):
            print(key, len(value))
        elif isinstance(value, str):
            print(key, value)


def test_dataset(dataset, with_3d=False, with_smpl=False):
    print('configs_yml:', args().configs_yml)
    print('model_version:', args().model_version)
    from models import smpl_model
    from visualization.visualization import Visualizer, draw_skeleton_multiperson
    test_projection_part = True if args().model_version in [4, 5, 6, 7] else False
    print('test_projection_part:', test_projection_part)

    save_dir = os.path.join(config.project_dir, 'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    print('Initialized dataset')

    batch_size, model_type = 2, 'smpl'
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, \
                            drop_last=False, pin_memory=False, num_workers=0)
    visualizer = Visualizer(resolution=(512, 512, 3), result_img_dir=save_dir, with_renderer=True)

    from visualization.visualization import make_heatmaps
    from utils.cam_utils import denormalize_cam_params_to_trans
    if with_smpl:
        smpl = smpl_model.create(args().smpl_model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path,
                                 J_reg_h36m17_path=args().smpl_J_reg_h37m_path, \
                                 batch_size=1, model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',
                                 flat_hand_mean=True, use_pca=False)
    img_size = 512
    bones, cm = constants.All54_connMat, constants.cm_All54

    for _, r in enumerate(dataloader):
        if _ % 100 == 0:
            print_data_shape(r)

        _check_minus2_error_(r['kp_3d'])

        for inds in range(2):
            img_bsname = os.path.basename(r['imgpath'][inds])
            image = r['image'][inds].numpy().astype(np.uint8)[:, :, ::-1]
            full_kp2d = (r['full_kp2d'][inds].numpy() + 1) * img_size / 2.0
            person_centers = (r['person_centers'][inds].numpy() + 1) * img_size / 2.0
            subject_ids = r['subject_ids'][inds]
            image_kp2d = draw_skeleton_multiperson(image.copy(), full_kp2d, bones=bones, cm=cm)

            for person_center, subject_id in zip(person_centers, subject_ids):
                y, x = person_center.astype(np.int)
                if y > 0 and x > 0:
                    image_kp2d[y - 10:y + 10, x - 10:x + 10] = [0, 0, 255]
                    cv2.putText(image_kp2d, 'id:{}'.format(subject_id), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 0, 255), 1)

            centermap_color = make_heatmaps(image.copy(), r['centermap'][inds])
            image_vis = np.concatenate([image_kp2d, centermap_color], 1)
            cv2.imwrite('{}/{}_{}_centermap.jpg'.format(save_dir, _, img_bsname), image_vis)
            if 'heatmap' in r:
                heatmap_color = make_heatmaps(image.copy(), r['heatmap'][inds])
                cv2.imwrite('{}/{}_{}_heatmap.jpg'.format(save_dir, _, img_bsname), heatmap_color)

            person_centers_onmap = ((r['person_centers'][inds].numpy() + 1) / 2.0 * (args().centermap_size - 1)).astype(
                np.int)
            positive_position = torch.stack(torch.where(r['centermap'][inds, 0] == 1)).permute(1, 0)

        if with_smpl and r['valid_masks'][0, 0, 4]:
            params, subject_ids = r['params'][0], r['subject_ids'][0]
            image = r['image_org'][0].numpy().astype(np.uint8)[:, :, ::-1]
            valid_mask = torch.where(r['valid_masks'][0, :, 4])[0]
            subject_ids = subject_ids[valid_mask]
            pose, betas = params[valid_mask][:, :66].float(), params[valid_mask][:, -10:].float()
            pose = torch.cat([pose, torch.zeros(len(pose), 6)], -1).float()
            output = smpl(poses=pose, betas=betas, get_skin=True)
            # if r['valid_masks'][0,0,6]:
            #     verts = r['verts'][0][r['valid_masks'][0,:,6]]
            # else:
            #     verts = output['verts']
            verts = output['verts']
            joints = output['j3d']

            verts[:, :, 2] += 2
            render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], images=image[None])[0]
            cv2.imwrite('{}/mesh_{}.png'.format(save_dir, _), render_img)
        j3ds = r['kp_3d'][0, 0]
        image = r['image_org'][0].numpy().astype(np.uint8)[:, :, ::-1]
        if r['valid_masks'][0, 0, 1]:
            pj2d = (j3ds[:, :2] + 1) * img_size / 2.0
            pj2d[j3ds[:, -1] == -2.] = -2.
            image_pkp3d = visualizer.draw_skeleton(image.copy(), pj2d, bones=bones, cm=cm)
            cv2.imwrite('{}/pkp3d_{}_{}.png'.format(save_dir, _, r['subject_ids'][0, 0]), image_pkp3d)
