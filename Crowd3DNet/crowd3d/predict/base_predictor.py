from ..base import *
from utils.demo_utils import convert_cam_to_3d_trans, save_meshes, get_video_bn, Time_counter
import platform, os
from utils.util import save_result_dict_tonpz
# from dataset.internet import img_preprocess

class Predictor(Base):
    def __init__(self):
        super(Predictor, self).__init__()
        self._build_model_()
        self._prepare_modules_()
        self.demo_cfg = {'mode': 'parsing', 'calc_loss': False}

    def net_forward(self, meta_data, cfg=None):
        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision == 'fp16':
            from torch.cuda.amp import autocast
            with autocast():
                outputs = self.model(meta_data, **cfg)
        else:
            outputs = self.model(meta_data, **cfg)
        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'],
                                                                                       outputs['reorganize_idx'])
        meta_data.update({'imgpath': imgpath_org, 'data_set': ds_org})
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org],
                                                                                             outputs[
                                                                                                 'reorganize_idx'].cpu().numpy())
        return outputs

    def _prepare_modules_(self):
        self.model.eval()
        self.demo_dir = os.path.join(config.project_dir, 'demo')

    def __initialize__(self):
        if self.save_visualization_on_img:
            self.visualizer = Visualizer(resolution=(512, 512), with_renderer=True)
        if self.save_mesh:
            self.smpl_faces = \
            pickle.load(open(os.path.join(args().smpl_model_path, 'SMPL_NEUTRAL.pkl'), 'rb'), encoding='latin1')['f']
        print('Initialization finished!')

    # def single_image_forward(self, image):
    #     meta_data = img_preprocess(image, '0', input_size=args().input_size, single_img_input=True)
    #     if '-1' not in self.gpu:
    #         meta_data['image'] = meta_data['image'].cuda()
    #     outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
    #     return outputs

    def reorganize_results(self, outputs, img_paths, reorganize_idx):
        if 'params' not in outputs:
            return {}, {}, {}
        results = {}
  
        # cam_results = outputs['params']['cam'].detach().cpu().numpy()
        smpl_pose_results = torch.cat([outputs['params']['global_orient'], outputs['params']['body_pose']],
                                      1).detach().cpu().numpy()
        smpl_shape_results = outputs['params']['betas'].detach().cpu().numpy()
        joints_54 = outputs['j3d'].detach().cpu().numpy()
        kp3d_smpl24_results = outputs['joints_smpl24'].detach().cpu().numpy()
        kp3d_spin24_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.SPIN_24)]
        kp3d_op25_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)]
        verts_results = outputs['verts'].detach().cpu().numpy()
        pj2d_results = outputs['pj2d'].detach().cpu().numpy()
        pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy()
        center_confs = outputs['centers_conf'].detach().cpu().numpy()
        verts_camed_results = outputs['verts_camed'].detach().cpu().numpy()
        centers_pred = outputs['centers_pred'].detach().cpu().numpy()
        hvip3d = outputs['hvip3d'].detach().cpu().numpy()
        final_trans=outputs['final_trans'].detach().cpu().numpy()
        tc3d = outputs['tc3d'].detach().cpu().numpy()
        # trans3d=outputs['pred_trans'].detach().cpu().numpy()


        vids_org = np.unique(reorganize_idx)
        center_dict = {}
        vertice_cam_dict = {}
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx == vid)[0]
            read_img_path = img_paths[verts_vids[0]]
            img_path = os.path.basename(read_img_path)
            results[img_path] = [{} for idx in range(len(verts_vids))]
            center_dict[img_path] = []
            vertice_cam_dict[img_path] = []
            cur_image = cv2.imread(read_img_path)
            cur_image_shape = cur_image.shape

            for subject_idx, batch_idx in enumerate(verts_vids):
                # results[img_path][subject_idx]['bbox_delta'] = cam_results[batch_idx]
                results[img_path][subject_idx]['pose'] = smpl_pose_results[batch_idx]
                results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
                results[img_path][subject_idx]['j3d_all54'] = joints_54[batch_idx]
                results[img_path][subject_idx]['j3d_smpl24'] = kp3d_smpl24_results[batch_idx]
                results[img_path][subject_idx]['j3d_spin24'] = kp3d_spin24_results[batch_idx]
                results[img_path][subject_idx]['j3d_op25'] = kp3d_op25_results[batch_idx]
                results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
                results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
                results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
                # results[img_path][subject_idx]['trans'] = convert_cam_to_3d_trans(cam_results[batch_idx])
                results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
                results[img_path][subject_idx]['verts_camed'] = verts_camed_results[batch_idx]

                results[img_path][subject_idx]['centers_pred'] = centers_pred[batch_idx]
                results[img_path][subject_idx]['hvip3d'] = hvip3d[batch_idx]
                results[img_path][subject_idx]['final_trans']=final_trans[batch_idx]
                results[img_path][subject_idx]['tc3d']=tc3d[batch_idx]
                
                cps = centers_pred[batch_idx]
                new_cps = np.around(cps * cur_image_shape[1] / 64).astype(int) # h == w
                new_cps = np.clip(new_cps, 0, cur_image_shape[1])
                center_dict[img_path].append(new_cps)
                vertice_cam_dict[img_path].append(verts_camed_results[batch_idx])
                results[img_path][subject_idx]['center']=new_cps

        return results, center_dict, vertice_cam_dict
