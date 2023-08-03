import numpy as np
import torch
import cv2
import torch.nn.functional as F
import trimesh
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy

import sys, os

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import constants
import config
from config import args
import utils.projection as proj
from utils.train_utils import process_idx, determine_rendering_order
from .renderer_pt3d import get_renderer
from pytorch3d.renderer import look_at_view_transform, get_world_to_view_transform
from .web_vis import write_to_html, convert_3dpose_to_line_figs, convert_image_list
from collections import OrderedDict

import pandas
import pickle

default_cfg = {'save_dir': None, 'vids': None, 'settings': []}  # 'put_org'


class Visualizer(object):
    def __init__(self, resolution=(512, 512), result_img_dir=None, with_renderer=False):
        self.resolution = resolution
        self.smpl_face = torch.from_numpy(
            pickle.load(open(os.path.join(args().smpl_model_path, 'SMPL_NEUTRAL.pkl'), 'rb'), \
                        encoding='latin1')['f'].astype(np.int32)).unsqueeze(0)
        if with_renderer:
            self.perps_proj = True  # args().perspective_proj
            T = None if self.perps_proj else torch.Tensor([[0., 0., 100]])
            self.renderer = get_renderer(resolution=self.resolution, perps=self.perps_proj, T=T)
        self.result_img_dir = result_img_dir
        self.heatmap_kpnum = 17
        self.vis_size = resolution
        self.mesh_color = (torch.Tensor([[[0.65098039, 0.74117647, 0.85882353]]]) * 255).long()
        self.color_table = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0], [128, 128, 0],
             [0, 128, 128], [128, 0, 128]])
        self.skeleton_3D_ploter = Plotter3dPoses()
        self.color_class_dict = {
            0: {0: [0.94, 1., 1.], 1: [0.49, 1., 0], 2: [0, 1., 1.], 3: [1., 0.98, 0.804], -1: [.9, .9, .8]}, \
            1: {0: [1., 0.753, 0.796], 1: [1, 0.647, 0], 2: [1, 0.431, 0.706], 3: [1., 0.98, 0.804], -1: [.9, .9, .8]}, \
            2: {0: [.9, .9, .8], 1: [.9, .9, .8], 2: [.9, .9, .8], 3: [.9, .9, .8], -1: [.9, .9, .8]}}

    def visualize_renderer_verts(self, verts, faces=None, image=None, cam_params=None, \
                                              color=torch.Tensor([.9, .9, .8]), trans=None, thresh=0., camK=None,
                                              patch_leftTop=None, img_pad_size=1):

        verts=verts.contiguous()
        if faces is None:
            faces = self.smpl_face.repeat(len(verts), 1, 1).to(verts.device)

        renderer = self.renderer


        if self.perps_proj:

            scale = torch.tensor(self.resolution, device=camK.device) / img_pad_size  * 2 / 3
            batch_fx = camK[:, 0, 0] * (1)
            batch_fy = camK[:, 1, 1] * (1)
            batch_cx_cy0 = torch.stack([camK[:, 0, 2], camK[:, 1, 2]], dim=-1)
            batch_cx_cy = batch_cx_cy0 + patch_leftTop * (-1)  # camK[:, :-1, -1]

            batch_fx_fy = torch.stack([batch_fx, batch_fy], dim=-1)
            # print('batch_fx_fy', batch_fx_fy)
            batch_fx_fy = batch_fx_fy * scale
            # print('batch_fx_fy', batch_fx_fy)
            batch_cx_cy = batch_cx_cy * scale #+ torch.tensor(self.resolution, device=camK.device)


        if trans is not None:
            verts += trans

        if self.perps_proj:
            rendered_img = renderer(verts, faces, colors=color, merge_meshes=True,
                                    focal_length=batch_fx_fy[0].unsqueeze(0),
                                    principal_point=batch_cx_cy[0].unsqueeze(0))
        else:
            rendered_img = None

        rendered_img = rendered_img.cpu().numpy().squeeze()

        if rendered_img.shape[-1] == 4:
            transparent = rendered_img[:, :, -1]
            rendered_img = rendered_img[:, :, :-1] * 255


        visible_weight = 0.9
        if image is not None:
            new_image = np.ones((self.resolution[0], self.resolution[1], 3), dtype=np.uint8) * 255
            new_image[:image.shape[0], :image.shape[1],:]=image

            valid_mask = (transparent > thresh)[:, :, np.newaxis]

            new_rendered_img = rendered_img * valid_mask * visible_weight + new_image * valid_mask * (
                    1 - visible_weight) + (
                                        1 - valid_mask) * new_image
        return new_rendered_img.astype(np.uint8)[:image.shape[0], :image.shape[1],:]
        
    # def visualize_renderer_verts_list(self, verts_list, faces_list=None, images=None, cam_params=None, \
    #                                   colors=torch.Tensor([.9, .9, .8]), trans=None, thresh=0.):
    #     verts_list = [verts.contiguous() for verts in verts_list]
    #     if faces_list is None:
    #         faces_list = [self.smpl_face.repeat(len(verts), 1, 1).to(verts.device) for verts in verts_list]
    #
    #     renderer = self.renderer
    #     rendered_imgs = []
    #     for ind, (verts, faces) in enumerate(zip(verts_list, faces_list)):
    #         if trans is not None:
    #             verts += trans[ind].unsqueeze(1)
    #
    #         color = colors[ind] if isinstance(colors, list) else colors
    #
    #         if self.perps_proj:
    #             rendered_img = renderer(verts, faces, colors=color, merge_meshes=True, cam_params=cam_params)
    #         else:
    #             verts[:, :, 2] -= 1.
    #             rendered_img = renderer(verts, faces, colors=color, merge_meshes=False, cam_params=cam_params)
    #             rendered_img = determine_rendering_order(rendered_img)
    #         rendered_imgs.append(rendered_img)
    #     rendered_imgs = torch.cat(rendered_imgs, 0).cpu().numpy()
    #     if rendered_imgs.shape[-1] == 4:
    #         transparent = rendered_imgs[:, :, :, -1]
    #         rendered_imgs = rendered_imgs[:, :, :, :-1] * 255
    #
    #     visible_weight = 0.9
    #     if images is not None:
    #         valid_mask = (transparent > thresh)[:, :, :, np.newaxis]
    #         rendered_imgs = rendered_imgs * valid_mask * visible_weight + images * valid_mask * (1 - visible_weight) + (
    #                 1 - valid_mask) * images
    #     return rendered_imgs.astype(np.uint8)

    def visualize_renderer_verts_list(self, verts_list, faces_list=None, images=None, cam_params=None, \
                                      colors=torch.Tensor([.9, .9, .8]), trans=None, thresh=0., camK=None,
                                      patch_leftTop=None, offsets=None):
        verts_list = [verts.contiguous() for verts in verts_list]
        if faces_list is None:
            faces_list = [self.smpl_face.repeat(len(verts), 1, 1).to(verts.device) for verts in verts_list]

        renderer = self.renderer
        rendered_imgs = []
        img_pad_size, crop_trbl, pad_trbl, data_scale = offsets[:, :2], offsets[:, 2:6], offsets[:, 6:10], offsets[:, 10].unsqueeze(-1)
        leftTop = torch.stack([crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]], 1)
        orign_size=img_pad_size * data_scale
        leftTop=leftTop * data_scale
        scale = torch.tensor(self.resolution, device=camK.device) / orign_size
        
        batch_fx = camK[:, 0, 0] * (1)
        batch_fy = camK[:, 1, 1] * (1)
        # batch_cx_cy = patch_leftTop * (-1)  # camK[:, :-1, -1]

        batch_cx_cy0 = torch.stack([camK[:, 0, 2], camK[:, 1, 2]], dim=-1) - leftTop
        batch_cx_cy = batch_cx_cy0 + patch_leftTop * (-1)  # camK[:, :-1, -1]

        batch_fx_fy = torch.stack([batch_fx, batch_fy], dim=-1)
        batch_fx_fy = batch_fx_fy * scale
        batch_cx_cy = batch_cx_cy * scale
        # 



        for ind, (verts, faces) in enumerate(zip(verts_list, faces_list)):
            if trans is not None:
                verts += trans[ind].unsqueeze(1)

            color = colors[ind] if isinstance(colors, list) else colors

            if self.perps_proj:
                rendered_img = renderer(verts, faces, colors=color, merge_meshes=True,
                                        focal_length=batch_fx_fy[ind].unsqueeze(0),
                                        principal_point=batch_cx_cy[ind].unsqueeze(0))
            else:
                rendered_img = None
            rendered_imgs.append(rendered_img)
        rendered_imgs = torch.cat(rendered_imgs, 0).cpu().numpy()
        if rendered_imgs.shape[-1] == 4:
            transparent = rendered_imgs[:, :, :, -1]
            rendered_imgs = rendered_imgs[:, :, :, :-1] * 255

        visible_weight = 0.9
        if images is not None:
            valid_mask = (transparent > thresh)[:, :, :, np.newaxis]
            rendered_imgs = rendered_imgs * valid_mask * visible_weight + images * valid_mask * (1 - visible_weight) + (
                    1 - valid_mask) * images
        return rendered_imgs.astype(np.uint8)

    def visualize_renderer_verts_list_overlap(self, verts_list, faces_list=None, images=None, cam_params=None, \
                                              colors=torch.Tensor([.9, .9, .8]), trans=None, thresh=0., camK=None,
                                              patch_leftTop=None, offsets=None):
        # fix 2x
        verts_list = [verts.contiguous() for verts in verts_list]
        if faces_list is None:
            faces_list = [self.smpl_face.repeat(len(verts), 1, 1).to(verts.device) for verts in verts_list]

        renderer = self.renderer
        rendered_imgs = []

        if self.perps_proj:
            img_pad_size, crop_trbl, pad_trbl, data_scale = offsets[:, :2], offsets[:, 2:6], offsets[:, 6:10], offsets[:, 10].unsqueeze(-1)
            leftTop = torch.stack([crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]], 1)
            orign_size=img_pad_size * data_scale
            leftTop=leftTop * data_scale
            # print('img_pad_size', img_pad_size.cpu().numpy())
            scale = torch.tensor(self.resolution, device=camK.device) / orign_size * 0.5
            batch_fx = camK[:, 0, 0] * (1)
            batch_fy = camK[:, 1, 1] * (1)
            batch_cx_cy0 = torch.stack([camK[:, 0, 2], camK[:, 1, 2]], dim=-1) - leftTop
            batch_cx_cy = batch_cx_cy0 + patch_leftTop * (-1)  # camK[:, :-1, -1]

            batch_fx_fy = torch.stack([batch_fx, batch_fy], dim=-1)
            # print('batch_fx_fy', batch_fx_fy)
            batch_fx_fy = batch_fx_fy * scale
            # print('batch_fx_fy', batch_fx_fy)
            batch_cx_cy = batch_cx_cy * scale + 0.25 * torch.tensor(self.resolution, device=camK.device)

        for ind, (verts, faces) in enumerate(zip(verts_list, faces_list)):
            if trans is not None:
                verts += trans[ind].unsqueeze(1)

            color = colors[ind] if isinstance(colors, list) else colors

            if self.perps_proj:
                rendered_img = renderer(verts, faces, colors=color, merge_meshes=True,
                                        focal_length=batch_fx_fy[ind].unsqueeze(0),
                                        principal_point=batch_cx_cy[ind].unsqueeze(0))
            else:
                rendered_img = None
            rendered_imgs.append(rendered_img)
        rendered_imgs = torch.cat(rendered_imgs, 0).cpu().numpy()
        if rendered_imgs.shape[-1] == 4:
            transparent = rendered_imgs[:, :, :, -1]
            rendered_imgs = rendered_imgs[:, :, :, :-1] * 255
            # 2x
            new_transparent, new_rendered_imgs = [], []
            new_size = self.resolution[0] * 2
            for i in range(transparent.shape[0]):
                new_transparent.append(cv2.resize(transparent[i], (new_size, new_size)))
                new_rendered_imgs.append(cv2.resize(rendered_imgs[i], (new_size, new_size)))
            new_transparent = np.array(new_transparent)
            new_rendered_imgs = np.array(new_rendered_imgs)

        visible_weight = 0.9

        if images is not None:
            new_images = np.ones((images.shape[0], images.shape[1] * 2, images.shape[1] * 2, 3), dtype=np.uint8) * 255
            start_h = start_w = self.resolution[0] // 2
            end_h = end_w = self.resolution[0] // 2 + self.resolution[0]
            for i in range(new_images.shape[0]):
                new_images[i, start_h:end_h, start_w:end_w] = images[i]
                # # the show of z
                verts_cur = verts_list[i]  # m, 6890, 3
                z = torch.mean(verts_cur[:, :, 2]).cpu().numpy()
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # new_images[i] = cv2.putText(new_images[i], str(z), (512, 512), font, 2, (0, 255, 0), 3)

            valid_mask = (new_transparent > thresh)[:, :, :, np.newaxis]
            # print('valid_mask sum', np.sum(valid_mask))
            new_rendered_imgs = new_rendered_imgs * valid_mask * visible_weight + new_images * valid_mask * (
                    1 - visible_weight) + (
                                        1 - valid_mask) * new_images
        return new_rendered_imgs.astype(np.uint8)

    def visulize_result(self, outputs, meta_data, show_items=['org_img', 'mesh'], vis_cfg=default_cfg, save2html=False,
                        **kwargs):
        vis_cfg = dict(default_cfg, **vis_cfg)
        if vis_cfg['save_dir'] is None:
            vis_cfg['save_dir'] = self.result_img_dir
            self.save_dir = self.result_img_dir

        else:
            self.save_dir = vis_cfg['save_dir']
        os.makedirs(vis_cfg['save_dir'], exist_ok=True)

        used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'], vids=vis_cfg['vids'])
        img_inds_org = [inds[0] for inds in per_img_inds]
        img_names = np.array(meta_data['imgpath'])[img_inds_org]
        org_imgs = meta_data['image'].cpu().numpy().astype(np.uint8)[img_inds_org]

        org_imgs=org_imgs[:, :, :, ::-1]
        # print('*****************org_imgs', org_imgs.shape)

        plot_dict = OrderedDict()
        for vis_name in show_items:
            org_img_figs = []
            if vis_name == 'org_img':
                if save2html:
                    plot_dict['org_img'] = {'figs': convert_image_list(org_imgs), 'type': 'image'}
                else:
                    plot_dict['org_img'] = {'figs': org_imgs, 'type': 'image'}

            if vis_name == 'mesh' and outputs['detection_flag']:
                per_img_verts_list = [outputs['verts_camed'][inds].detach() for inds in per_img_inds]
                camK = meta_data['camK'].to(outputs['verts_camed'].device)[img_inds_org]
                patch_leftTop = meta_data['patch_leftTop'].to(outputs['verts_camed'].device)[img_inds_org]
                offsets = meta_data['offsets'].to(outputs['verts_camed'].device)[img_inds_org]
                rendered_imgs_overlap=None
                rendered_imgs=None
                rendered_imgs = self.visualize_renderer_verts_list(per_img_verts_list, images=org_imgs.copy(),
                                                                   camK=camK, patch_leftTop=patch_leftTop,
                                                                   offsets=offsets)
                # rendered_imgs_overlap = self.visualize_renderer_verts_list_overlap(per_img_verts_list,
                #                                                                    images=org_imgs.copy(),
                #                                                                    camK=camK,
                #                                                                    patch_leftTop=patch_leftTop,
                #                                                                    offsets=offsets)
                padding=False

                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = org_imgs[img_id].copy()
                    h, w, _ = org_img.shape
                    padding_w = int(w * 0.5)
                    padding_h = int(h * 0.5)
                    new_h = h + 2 * padding_h
                    new_w = w + 2 * padding_w

                    old_hvip2d = outputs['old_hvip2d'][inds_list]
                    old_list=[]
                    for old in old_hvip2d:  # outputs['pj2d'][inds_list]
                        old_org = ((old + 1) / 2 * org_imgs.shape[1])  # w,h
                        if padding:
                            rendered_imgs_overlap[img_id] = cv2.circle(rendered_imgs_overlap[img_id],
                                                            (int(old_org[0] + padding_w), int(old_org[1] + padding_h)),
                                                            3, (255, 245, 0), 2)
                        else:
                            rendered_imgs[img_id] = cv2.circle(rendered_imgs[img_id],
                                                            (int(old_org[0]), int(old_org[1])),
                                                            3, (255, 245, 0), 2)

                        old_list.append(old_org)
                    torso_center=outputs['torso_center'][per_img_inds[img_id]]
                    torso_center=((torso_center + 1) / 2 * org_imgs.shape[1])
                    body_center=outputs['centers_pred'][per_img_inds[img_id]]/ args().centermap_size * org_imgs.shape[1]
                    for pid in range(torso_center.shape[0]):
                        tc=torso_center[pid]
                        if padding:
                            rendered_imgs_overlap[img_id] = cv2.circle(rendered_imgs_overlap[img_id],
                                                            (int(tc[0] + padding_w), int(tc[1] + padding_h)),
                                                            3, (147, 20, 255), 2)
                            old_org=old_list[pid]
                            rendered_imgs_overlap[img_id] = cv2.line( rendered_imgs_overlap[img_id], (int(old_org[0] + padding_w), int(old_org[1] + padding_h) ), (int(tc[0] + padding_w), int(tc[1] + padding_h)), color=(102, 126, 139), thickness=2)
                        else:
                            rendered_imgs[img_id] = cv2.circle(rendered_imgs[img_id],
                                                        (int(tc[0]), int(tc[1])),
                                                        3, (147, 20, 255), 2)
                        old_org=old_list[pid]
                        rendered_imgs[img_id] = cv2.line( rendered_imgs[img_id], (int(old_org[0]), int(old_org[1]) ), (int(tc[0]), int(tc[1])), color=(102, 126, 139), thickness=2)
  


                # print('rendered_imgs', rendered_imgs.shape, rendered_imgs.dtype)
                if 'put_org' in vis_cfg['settings']:
                    offsets = meta_data['offsets'].cpu().numpy().astype(np.int)[img_inds_org]
                    img_pad_size, crop_trbl, pad_trbl = offsets[:, :2], offsets[:, 2:6], offsets[:, 6:10]
                    rendering_onorg_images = []
                    for inds, j in enumerate(used_org_inds):
                        org_imge = cv2.imread(img_names[inds])
                        (ih, iw), (ph, pw) = org_imge.shape[:2], img_pad_size[inds]
                        resized_images = cv2.resize(rendered_imgs[inds], (ph + 1, pw + 1),
                                                    interpolation=cv2.INTER_CUBIC)
                        (ct, cr, cb, cl), (pt, pr, pb, pl) = crop_trbl[inds], pad_trbl[inds]
                        org_imge[ct:ih - cb, cl:iw - cr] = resized_images[pt:ph - pb, pl:pw - pr]
                        rendering_onorg_images.append(org_imge)
                    if save2html:
                        plot_dict['mesh_rendering_orgimgs'] = {'figs': convert_image_list(rendering_onorg_images),
                                                               'type': 'image'}
                    else:
                        plot_dict['mesh_rendering_orgimgs'] = {'figs': rendering_onorg_images, 'type': 'image'}

                if save2html:
                    plot_dict['mesh_rendering_imgs'] = {'figs': convert_image_list(rendered_imgs), 'type': 'image'}
                else:
                    plot_dict['mesh_rendering_imgs'] = {'figs': rendered_imgs, 'type': 'image'}
                    # plot_dict['mesh_rendering_imgs_overlap'] = {'figs': rendered_imgs_overlap, 'type': 'image'}

            if vis_name == 'j3d' and outputs['detection_flag']:
                real_aligned, pred_aligned, pos3d_vis_mask, joint3d_bones = kwargs['kp3ds']
                real_3ds = (real_aligned * pos3d_vis_mask.unsqueeze(-1)).cpu().numpy()
                predicts = (pred_aligned * pos3d_vis_mask.unsqueeze(-1)).detach().cpu().numpy()
                if save2html:
                    plot_dict['j3d'] = {'figs': convert_3dpose_to_line_figs([predicts, real_3ds], joint3d_bones),
                                        'type': 'skeleton'}
                else:
                    skeleton_3ds = []
                    for inds in per_img_inds:
                        for real_pose_3d, pred_pose_3d in zip(real_3ds[inds], predicts[inds]):
                            skeleton_3d = self.skeleton_3D_ploter.encircle_plot([real_pose_3d, pred_pose_3d], \
                                                                                joint3d_bones,
                                                                                colors=[(255, 0, 0), (0, 255, 255)])
                            skeleton_3ds.append(skeleton_3d)
                    plot_dict['j3d'] = {'figs': np.array(skeleton_3ds), 'type': 'skeleton'}

            if vis_name == 'pj2d' and outputs['detection_flag']:
                kp_imgs = []
                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = org_imgs[img_id].copy()
                    try:
                        if 'full_kp2d' in meta_data:
                            real_kp2d = meta_data['full_kp2d'].to(outputs['pj2d'].device)[inds_list]
                        for i, kp2d_vis in enumerate(outputs['pj2d_backup'][inds_list][:, :54]):  # outputs['pj2d'][inds_list]

                            if len(kp2d_vis) > 0:
                                kp2d_vis_org = ((kp2d_vis + 1) / 2 * org_imgs.shape[1])  # 54, 2
                                kp2d_vis_org_temp = kp2d_vis_org.clone()
                                if 'full_kp2d' in meta_data:
                                    real_i = real_kp2d[i]
                                    valid = real_i[:, 0] > -2
                                    kp2d_vis_org_temp[~valid] = -2

                                # org_img = draw_skeleton(org_img, kp2d_vis, bones=constants.body17_connMat, cm=constants.cm_body17)
                                org_img = draw_skeleton(org_img, kp2d_vis_org_temp, bones=constants.All54_connMat,
                                                        cm=constants.cm_All54)
                    except Exception as error:
                        print(error, ' reported while drawing 2D pose')
                    kp_imgs.append(org_img)
                if save2html:
                    kp_imgs = convert_image_list(kp_imgs)
                plot_dict['pj2d'] = {'figs': kp_imgs, 'type': 'image'}

            if vis_name == 'pj2d_overlap' and outputs['detection_flag']:
                kp_imgs = []
                padding_ratio = 0.5
                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = org_imgs[img_id].copy()
                    h, w, _ = org_img.shape
                    padding_w = int(w * padding_ratio)
                    padding_h = int(h * padding_ratio)
                    new_h = h + 2 * padding_h
                    new_w = w + 2 * padding_w
                    patch_image_padded = np.ones((new_h, new_w, 3), dtype=org_img.dtype) * 255
                    patch_image_padded[padding_h:padding_h + h, padding_w:padding_w + w, :] = org_img

                    try:
                        if 'full_kp2d' in meta_data:
                            real_kp2d = meta_data['full_kp2d'].to(outputs['pj2d'].device)[inds_list]
                        for i, kp2d_vis in enumerate(outputs['pj2d_backup'][inds_list][:, :54]):  # outputs['pj2d'][inds_list]
                            if len(kp2d_vis) > 0:
                                kp2d_vis_org = ((kp2d_vis + 1) / 2 * org_imgs.shape[1])
                                kp2d_vis_org_temp = kp2d_vis_org.clone()
                                if 'full_kp2d' in meta_data:
                                    real_i = real_kp2d[i]
                                    valid = real_i[:, 0] > -2
                                    kp2d_vis_org_temp[~valid] = -2
                                # org_img = draw_skeleton(org_img, kp2d_vis, bones=constants.body17_connMat, cm=constants.cm_body17)
                                patch_image_padded = draw_skeleton(patch_image_padded, kp2d_vis_org_temp,
                                                                   bones=constants.All54_connMat,
                                                                   cm=constants.cm_All54, padding_size=(
                                        padding_w, padding_h))  # , padding_size=(padding_w, padding_h)
                        if 'full_bbox' in meta_data:
                            for bbox_i in outputs['full_bbox'][inds_list]:
                                color_i=np.random.rand(3)*255
                                bbox_i_org = ((bbox_i + 1) / 2 * org_imgs.shape[1])
                                bbox_i_org = bbox_i_org.cpu().detach().numpy().astype(int)
                                patch_image_padded = cv2.rectangle(patch_image_padded,
                                                                   (bbox_i_org[0] + padding_w, bbox_i_org[1] + padding_h),
                                                                   (bbox_i_org[2] + padding_w, bbox_i_org[3] + padding_h),
                                                                   color_i, thickness=2)


                    except Exception as error:
                        print(error, ' reported while drawing 2D pose')
                    kp_imgs.append(patch_image_padded)

                plot_dict['pj2d_overlap'] = {'figs': kp_imgs, 'type': 'image'}

            if vis_name == 'gt_check':
                kp_imgs = []
                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = org_imgs[img_id].copy()
                    try:
                        real_kp2d = meta_data['full_kp2d'].to(outputs['pj2d'].device)[inds_list]
                        # real_bbox = meta_data['full_bbox2d'].to(outputs['pj2d'].device)[inds_list]
                        # assert len(real_kp2d) == len(real_bbox), 'len(real_kp2d)!=len(real_bbox)'
                        for id, kp2d_vis in enumerate(real_kp2d):  # outputs['pj2d'][inds_list]

                            if len(kp2d_vis) > 0:
                                kp2d_vis = ((kp2d_vis + 1) / 2 * org_imgs.shape[1])
                                # org_img = draw_skeleton(org_img, kp2d_vis, bones=constants.body17_connMat, cm=constants.cm_body17)
                                org_img = draw_skeleton(org_img, kp2d_vis, bones=constants.All54_connMat,
                                                        cm=constants.cm_All54)
                            # bbox_i = real_bbox[id]  # 4,
                            # bbox_i = (bbox_i + 1) / 2 * org_imgs.shape[1]
                            # bbox_i = bbox_i.cpu().detach().numpy().astype(int)
                            # org_img = cv2.rectangle(org_img, (bbox_i[0], bbox_i[1]), (bbox_i[2], bbox_i[3]),
                            #                         (0, 255, 255), thickness=2)
                        real_person_centers=meta_data['person_centers'].to(outputs['pj2d'].device)[inds_list]
                        for person_center in real_person_centers:
                            person_center = ((person_center + 1) / 2 * org_imgs.shape[1])
                            org_img = cv2.circle(org_img,(int(person_center[1]), int(person_center[0])),
                                                            5, (200, 255, 0), 2)
                    except Exception as error:
                        print(error, ' reported while drawing 2D pose')
                    kp_imgs.append(org_img)

                plot_dict['gt_check'] = {'figs': kp_imgs, 'type': 'image'}

            if vis_name == 'hvip2d' and outputs['detection_flag']:
                hvip_imgs = []
                padding_ratio = 0.5

                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = org_imgs[img_id].copy()
                    h, w, _ = org_img.shape
                    padding_w = int(w * padding_ratio)
                    padding_h = int(h * padding_ratio)
                    new_h = h + 2 * padding_h
                    new_w = w + 2 * padding_w
                    patch_image_padded = np.ones((new_h, new_w, 3), dtype=org_img.dtype) * 255
                    patch_image_padded[padding_h:padding_h + h, padding_w:padding_w + w, :] = org_img
                    try:
                        old_hvip2d = outputs['old_hvip2d'][inds_list]
                        new_hvip2d = outputs['new_hvip2d'][inds_list]
    
                        for old, new in zip(old_hvip2d, new_hvip2d):  # outputs['pj2d'][inds_list]
                            old_org = ((old + 1) / 2 * org_imgs.shape[1])  # w,h
                            new_org = ((new + 1) / 2 * org_imgs.shape[1])
                            patch_image_padded = cv2.circle(patch_image_padded,
                                                            (int(old_org[0] + padding_w), int(old_org[1] + padding_h)),
                                                            3, (255, 0, 0), -1)
                            patch_image_padded = cv2.circle(patch_image_padded,
                                                            (int(new_org[0] + padding_w), int(new_org[1] + padding_h)),
                                                            3, (255, 255, 0), 2)
                        if 'hvip2ds' in meta_data:
                            real_hvip2ds = meta_data['hvip2ds'].to(outputs['old_hvip2d'].device)[inds_list]
                            for real_hvip2d in real_hvip2ds:
                                real_hvip2d_org=((real_hvip2d + 1) / 2 * org_imgs.shape[1])
                                patch_image_padded = cv2.circle(patch_image_padded,
                                                                (int(real_hvip2d_org[0] + padding_w), int(real_hvip2d_org[1] + padding_h)),
                                                                3, (125, 255, 0), -1)
                        if 'predict_ankle2d' in outputs:
                            ankle2ds=outputs['predict_ankle2d'][inds_list]
        
                            for ankle2d in ankle2ds:
                                ankle2d_org=((ankle2d + 1) / 2 * org_imgs.shape[1]) # 4,
                                patch_image_padded = cv2.circle(patch_image_padded,
                                                            (int(ankle2d_org[0][0] + padding_w), int(ankle2d_org[0][1] + padding_h)),
                                                            2, (100, 50, 0), 2)
                                patch_image_padded = cv2.circle(patch_image_padded,
                                                            (int(ankle2d_org[1][0] + padding_w), int(ankle2d_org[1][1] + padding_h)),
                                                            2, (100, 50, 0), 2)
                    except Exception as error:
                        print(error, ' reported while drawing hvip2d')
                    hvip_imgs.append(patch_image_padded)
                if save2html:
                    hvip_imgs = convert_image_list(hvip_imgs)
                plot_dict['hvip2d'] = {'figs': hvip_imgs, 'type': 'image'}

            if vis_name == 'hp_aes' and outputs['detection_flag']:
                heatmaps_AEmaps = []
                # hp_aes = torch.nn.functional.interpolate(hp_aes[vids],size=(img_size,img_size),mode='bilinear',align_corners=True)
                for img_id, hp_ae in enumerate(outputs['kp_ae_maps'][used_org_inds]):
                    img_bk = cv2.resize(org_imgs[img_id].copy(), (hp_ae.shape[1], hp_ae.shape[2]))
                    heatmaps_AEmaps.append(np.vstack([make_heatmaps(img_bk, hp_ae[:self.heatmap_kpnum]),
                                                      make_tagmaps(img_bk, hp_ae[self.heatmap_kpnum:])]))

            if vis_name == 'centermap' and outputs['detection_flag']:
                centermaps_list = []
                for img_id, centermap in enumerate(outputs['center_map'][used_org_inds]):
                    img_bk = cv2.resize(org_imgs[img_id].copy(), org_imgs.shape[1:3])
                    # if 'torso_center' in outputs:
                    #     torso_center=outputs['torso_center'][per_img_inds[img_id]]
                    #     torso_center=((torso_center + 1) / 2 * org_imgs.shape[1])
                    #     for tc in torso_center:
                    #         cv2.drawMarker(img_bk, position=(int(tc[0]), int(tc[1])),color=(0, 0, 255),markerSize = 10, markerType=cv2.MARKER_CROSS, thickness=5)
                    res=make_heatmaps(img_bk, centermap)
                    res_heatmap=res[:, 512:, :]
                    padding=True
                    if padding:
                        temp=np.ones((712, 712, 3))*255
                        temp[100:612, 100:612, :]=res_heatmap
                        res_heatmap=temp

                    if 'torso_center' in outputs and 'centers_pred' in outputs:
                        torso_center=outputs['torso_center'][per_img_inds[img_id]]
                        torso_center=((torso_center + 1) / 2 * org_imgs.shape[1])

                        body_center=outputs['centers_pred'][per_img_inds[img_id]]/ args().centermap_size * org_imgs.shape[1]

                        for pid in range(torso_center.shape[0]):
                            tc=torso_center[pid]
                            bc=body_center[pid]

                            if padding:
                                cv2.circle(res_heatmap, (int(tc[0]+100), int(tc[1]+100)), 3, (147, 20, 255), 2)
                                # cv2.arrowedLine(res_heatmap, (int(bc[0]+100), int(bc[1]+100)), (int(tc[0]+100), int(tc[1]+100)),(255, 0, 255), 2, tipLength=0.15)
                            else:
                                cv2.arrowedLine(res_heatmap, (int(bc[0]), int(bc[1])), (int(tc[0]), int(tc[1])),(0, 0, 255), thickness=1, tipLength=0.1)
                                

                    
                    centermaps_list.append(res_heatmap)
                if save2html:
                    centermaps_list = convert_image_list(centermaps_list)
                plot_dict['centermap'] = {'figs': centermaps_list, 'type': 'image'}

        if save2html:
            write_to_html(img_names, plot_dict, vis_cfg)
        self.write_image(plot_dict, img_names, kwargs['global_count'] if 'global_count' in kwargs else '')
        return plot_dict, img_names

    def draw_skeleton(self, image, pts, **kwargs):
        return draw_skeleton(image, pts, **kwargs)

    def draw_skeleton_multiperson(self, image, pts, **kwargs):
        return draw_skeleton_multiperson(image, pts, **kwargs)

    # created by wh
    def write_image(self, plot_dict, img_names, global_count):
        # print('visual results in %s' % self.result_img_dir)
        # print(img_names)
        # print(plot_dict.keys())
        if 'org_img' in plot_dict:
            org_img = plot_dict['org_img']['figs']
        else:
            org_img = None

        if 'pj2d' in plot_dict:
            pj2d = plot_dict['pj2d']['figs']
        else:
            pj2d = None

        if 'centermap' in plot_dict:
            centermap = plot_dict['centermap']['figs']
        else:
            centermap = None

        if 'gt_check' in plot_dict:
            gt_check = plot_dict['gt_check']['figs']
        else:
            gt_check = None

        if 'pj2d_overlap' in plot_dict:
            pj2d_overlap = plot_dict['pj2d_overlap']['figs']
        else:
            pj2d_overlap = None

        if 'mesh_rendering_imgs' in plot_dict:
            mesh_rendering_imgs = plot_dict['mesh_rendering_imgs']['figs']
        else:
            mesh_rendering_imgs = None

        if 'mesh_rendering_orgimgs' in plot_dict:
            mesh_rendering_orgimgs = plot_dict['mesh_rendering_orgimgs']['figs']
        else:
            mesh_rendering_orgimgs = None

        if 'mesh_rendering_imgs_overlap' in plot_dict:
            mesh_rendering_imgs_overlap = plot_dict['mesh_rendering_imgs_overlap']['figs']
        else:
            mesh_rendering_imgs_overlap = None

        if 'hvip2d' in plot_dict:
            hvip2d = plot_dict['hvip2d']['figs']
        else:
            hvip2d = None

        if 'j3d' in plot_dict:
            j3d = plot_dict['j3d']['figs']
        else:
            j3d = None

        for i in range(img_names.shape[0]):
            img_name = img_names[i]
            img_name = str(img_name).split('/')[-1]
            temp=''
            if global_count != '':
                temp='step'+str(global_count).zfill(10)+'_'
            save_path = os.path.join(self.save_dir, temp +
                                     img_name.replace('.jpg', '').replace('.png', ''))
            os.makedirs(save_path, exist_ok=True)
            if org_img is not None:
                cv2.imwrite(os.path.join(save_path, 'org.jpg'), org_img[i])
            if pj2d is not None:
                cv2.imwrite(os.path.join(save_path, 'pj2d.jpg'), pj2d[i])
            if centermap is not None:
                cv2.imwrite(os.path.join(save_path, 'centermap.jpg'), centermap[i])
            if gt_check is not None:
                cv2.imwrite(os.path.join(save_path, 'gt_check.jpg'), gt_check[i])
            if pj2d_overlap is not None:
                cv2.imwrite(os.path.join(save_path, 'pj2d_overlap.jpg'), pj2d_overlap[i])
            if mesh_rendering_imgs is not None:
                cv2.imwrite(os.path.join(save_path, 'mesh_rendering_imgs.jpg'), mesh_rendering_imgs[i])
            if mesh_rendering_orgimgs is not None:
                cv2.imwrite(os.path.join(save_path, 'mesh_rendering_orgimgs.jpg'), mesh_rendering_orgimgs[i])
            if mesh_rendering_imgs_overlap is not None:
                cv2.imwrite(os.path.join(save_path, 'mesh_rendering_imgs_overlap.jpg'), mesh_rendering_imgs_overlap[i])
            if hvip2d is not None:
                cv2.imwrite(os.path.join(save_path, 'hvip2d.jpg'), hvip2d[i])
            if j3d is not None:
                cv2.imwrite(os.path.join(save_path, 'j3d.jpg'), j3d[i])


def make_heatmaps(image, heatmaps):
    heatmaps = torch.nn.functional.interpolate(heatmaps[None], size=image.shape[:2], mode='bilinear')[0]
    heatmaps = heatmaps.mul(255) \
        .clamp(0, 255) \
        .byte() \
        .detach().cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_grid = np.zeros((height, (num_joints + 1) * width, 3), dtype=np.uint8)

    for j in range(num_joints):
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap * 0.7 + image * 0.3

        width_begin = width * (j + 1)
        width_end = width * (j + 2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image

    return image_grid

def make_heatmaps_temp(image, image_bk, heatmaps):
    heatmaps = torch.nn.functional.interpolate(heatmaps[None], size=image.shape[:2], mode='bilinear')[0]
    heatmaps = heatmaps.mul(255) \
        .clamp(0, 255) \
        .byte() \
        .detach().cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_grid = np.zeros((height, (num_joints + 1) * width, 3), dtype=np.uint8)

    for j in range(num_joints):
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap * 0.7 + image * 0.3

        width_begin = width * (j + 1)
        width_end = width * (j + 2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image

    return image_grid

def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints + 1) * width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.add(-min) \
            .div(max - min + 1e-5) \
            .mul(255) \
            .clamp(0, 255) \
            .byte() \
            .detach().cpu().numpy()

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap * 0.9 + image_resized * 0.1

        width_begin = width * (j + 1)
        width_end = width * (j + 2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def draw_skeleton(image, pts, bones=None, cm=None, label_kp_order=False, r=3, padding_size=None):
    pts = pts.clone()
    pts = pts.cpu().detach().numpy()
    for i, pt in enumerate(pts):
        if len(pt) > 1:
            if pt[0] > 0 and pt[1] > 0:
                if padding_size is not None:
                    pt[0] = int(pt[0])
                    pt[1] = int(pt[1])
                    padding_w, padding_h = padding_size
                    pt[0] = pt[0] + padding_w
                    pt[1] = pt[1] + padding_h

                if i in [10, 11]:
                    image = cv2.circle(image, (int(pt[0]), int(pt[1])), r, (0, 0, 0), -1)
                else:
                    image = cv2.circle(image, (int(pt[0]), int(pt[1])), r, (255, 0, 0), -1)
                if label_kp_order and i in bones:
                    img = cv2.putText(image, str(i), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                      (255, 215, 0), 1)

    if bones is not None:
        if cm is None:
            set_colors = np.array([[255, 0, 0] for i in range(len(bones))]).astype(np.int)
        else:
            if len(bones) > len(cm):
                cm = np.concatenate([cm for _ in range(len(bones) // len(cm) + 1)], 0)
            set_colors = cm[:len(bones)].astype(np.int)
        bones = np.concatenate([bones, set_colors], 1).tolist()
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa > 0).all() and (pb > 0).all():
                xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
                image = cv2.line(image, (xa, ya), (xb, yb), (int(line[2]), int(line[3]), int(line[4])), r)
    return image


def draw_skeleton_multiperson(image, pts_group, **kwargs):
    for pts in pts_group:
        image = draw_skeleton(image, pts, **kwargs)
    return image


class Plotter3dPoses:

    def __init__(self, canvas_size=(512, 512), origin=(0.5, 0.5), scale=200):
        self.canvas_size = canvas_size
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta, self.phi = 0, np.pi / 2  # np.pi/4, -np.pi/6
        axis_length = 200
        axes = [
            np.array([[-axis_length / 2, -axis_length / 2, 0], [axis_length / 2, -axis_length / 2, 0]],
                     dtype=np.float32),
            np.array([[-axis_length / 2, -axis_length / 2, 0], [-axis_length / 2, axis_length / 2, 0]],
                     dtype=np.float32),
            np.array([[-axis_length / 2, -axis_length / 2, 0], [-axis_length / 2, -axis_length / 2, axis_length]],
                     dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8) * 255 if img is None else img
        R = self._get_rotation(self.theta, self.phi)
        # self._draw_axes(img, R)
        for vertices, color in zip(pose_3ds, colors):
            self._plot_edges(img, vertices, bones, R, color)
        return img

    def encircle_plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8) * 255 if img is None else img
        # encircle_theta, encircle_phi = [0, np.pi/4, np.pi/2, 3*np.pi/4], [np.pi/2,np.pi/2,np.pi/2,np.pi/2]
        encircle_theta, encircle_phi = [0, 0, 0, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 2, np.pi / 2], [
            np.pi / 2, 5 * np.pi / 7, -2 * np.pi / 7, np.pi / 2, 5 * np.pi / 7, -2 * np.pi / 7, np.pi / 2,
            5 * np.pi / 7, -2 * np.pi / 7, ]
        encircle_origin = np.array([[0.165, 0.165], [0.165, 0.495], [0.165, 0.825], \
                                    [0.495, 0.165], [0.495, 0.495], [0.495, 0.825], \
                                    [0.825, 0.165], [0.825, 0.495], [0.825, 0.825]], dtype=np.float32) * \
                          np.array(self.canvas_size)[None]
        for self.theta, self.phi, self.origin in zip(encircle_theta, encircle_phi, encircle_origin):
            R = self._get_rotation(self.theta, self.phi)
            # self._draw_axes(img, R)
            for vertices, color in zip(pose_3ds, colors):
                self._plot_edges(img, vertices * 0.6, bones, R, color)
        return img

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R, color):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        org_verts = vertices.reshape((-1, 3))[edges]
        for inds, edge_vertices in enumerate(edges_vertices):
            if 0 in org_verts[inds]:
                continue
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), color, 2, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [cos(theta), sin(theta) * sin(phi)],
            [-sin(theta), cos(theta) * sin(phi)],
            [0, -cos(phi)]
        ], dtype=np.float32)  # transposed


def test_visualizer():
    visualizer = Visualizer(resolution=(512, 512), input_size=args().input_size, result_img_dir=args().result_img_dir,
                            with_renderer=True)




if __name__ == '__main__':
    test_visualizer()
