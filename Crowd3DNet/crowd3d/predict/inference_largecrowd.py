import os

from .base_predictor import *
import constants
import glob
from utils.util import collect_image_list
import pickle

class Image_processor(Predictor):
    def __init__(self):
        super(Image_processor, self).__init__()
        self.__initialize__()
        
    def reorganize_results(self, outputs, img_paths, reorganize_idx):
        if 'params' not in outputs:
            return {} #, {}, {} # wh modify at 7.26.
        results = {}
        # cam_results = outputs['params']['cam'].detach().cpu().numpy()
        smpl_pose_results = torch.cat([outputs['params']['global_orient'], outputs['params']['body_pose']],
                                    1).detach().cpu().numpy()
        smpl_shape_results = outputs['params']['betas'].detach().cpu().numpy()
        pj2d_results = outputs['pj2d'].detach().cpu().numpy()
        center_confs = outputs['centers_conf'].detach().cpu().numpy()
        centers_pred = outputs['centers_pred'].detach().cpu().numpy()
        hvip3d = outputs['hvip3d'].detach().cpu().numpy()
        final_trans=outputs['final_trans'].detach().cpu().numpy()
        hvip2d_org=outputs['hvip2d_org'].detach().cpu().numpy()
        hvip2d_in_scene=outputs['hvip2d_in_scene'].detach().cpu().numpy()
        tc3d = outputs['tc3d'].detach().cpu().numpy()
        torso_center=outputs['torso_center'].detach().cpu().numpy()
        # predict_ankle2d=outputs['predict_ankle2d'].detach().cpu().numpy()

        # too big
        # verts_camed_results = outputs['verts_camed'].detach().cpu().numpy()
        # joints_54 = outputs['j3d'].detach().cpu().numpy()
        # kp3d_smpl24_results = outputs['joints_smpl24'].detach().cpu().numpy()
        # kp3d_spin24_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.SPIN_24)]
        # kp3d_op25_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)]
        # verts_results = outputs['verts'].detach().cpu().numpy()
        pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy()

        vids_org = np.unique(reorganize_idx)
        center_dict = {}
        vertice_cam_dict = {}
        data_scale=outputs['meta_data']['offsets'][:, 10]

        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx == vid)[0]
            read_img_path = img_paths[verts_vids[0]]
            img_path = os.path.basename(read_img_path)
            results[img_path] = [{} for idx in range(len(verts_vids))]
            center_dict[img_path] = []
            vertice_cam_dict[img_path] = []

            cur_image=cv2.imread(read_img_path)
            cur_data_scale=data_scale[verts_vids[0]]
            patch_size=cur_image.shape[0]*cur_data_scale
            patch_size=int(patch_size.detach().cpu())

            for subject_idx, batch_idx in enumerate(verts_vids):
                results[img_path][subject_idx]['pose'] = smpl_pose_results[batch_idx]
                results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
                results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
                results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
                results[img_path][subject_idx]['centers_pred'] = centers_pred[batch_idx]
                results[img_path][subject_idx]['hvip3d'] = hvip3d[batch_idx]
                results[img_path][subject_idx]['final_trans']=final_trans[batch_idx]
                results[img_path][subject_idx]['hvip2d_org']=hvip2d_org[batch_idx]
                results[img_path][subject_idx]['hvip2d_in_scene']=hvip2d_in_scene[batch_idx]
                results[img_path][subject_idx]['tc3d']=tc3d[batch_idx]

                # results[img_path][subject_idx]['predict_ankle2d']=predict_ankle2d[batch_idx]

                # too big
                # results[img_path][subject_idx]['j3d_all54'] = joints_54[batch_idx]
                # results[img_path][subject_idx]['j3d_smpl24'] = kp3d_smpl24_results[batch_idx]
                # results[img_path][subject_idx]['j3d_spin24'] = kp3d_spin24_results[batch_idx]
                # results[img_path][subject_idx]['j3d_op25'] = kp3d_op25_results[batch_idx]
                # results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
                results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
                # results[img_path][subject_idx]['verts_camed'] = verts_camed_results[batch_idx]
                
                cps = centers_pred[batch_idx] # 0,64
                new_cps = np.around(cps * patch_size / 64).astype(int) # h == w
                new_cps = np.clip(new_cps, 0, patch_size)
                center_dict[img_path].append(new_cps)
                results[img_path][subject_idx]['center']=new_cps

                tcs= torso_center[batch_idx] # -1,1
                tcs=(tcs+1)/2 # (0,1)
                new_tcs=np.around(tcs*patch_size).astype(int)
                results[img_path][subject_idx]['torso_center']=new_tcs
        return results

    @torch.no_grad()
    def run(self, data_root, ground_cam_root, scene_type, test_dataset):
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer.result_img_dir = os.path.join(self.output_dir, 'images_results')
        counter = Time_counter(thresh=1)

        internet_loader = self._create_single_data_loader(dataset=test_dataset, train_flag=False, data_root=data_root, ground_cam_root=ground_cam_root, scene_type=scene_type,
                                                          shuffle=False)
        counter.start()

        all_result={}
        for test_iter, meta_data in enumerate(internet_loader):
            outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)
            all_result.update(results)

            # if self.save_dict_results:
            #     save_result_dict_tonpz(results, os.path.join(self.output_dir, 'results'))

            if self.save_visualization_on_img:
                show_items_list = ['mesh','org_img', 'pj2d', 'pj2d_overlap', 'hvip'] #['org_img', 'mesh']
                if self.save_centermap:
                    show_items_list.append('centermap')
                results_dict, img_names = self.visualizer.visulize_result(outputs, outputs['meta_data'], \
                                                                          show_items=show_items_list,
                                                                          vis_cfg={},
                                                                          save2html=False)
            counter.start()
        print('Processed %d images, saved in %s' %(len(internet_loader.dataset), self.output_dir))
        #np.savez(os.path.join(self.output_dir, 'all_result.npz'), results=all_result)
        with open(os.path.join(self.output_dir, 'all_result.pkl'), 'wb') as wf:
            pickle.dump(all_result, wf)

def main():
    input_args = sys.argv[1:]
    if sum(['configs_yml' in input_arg for input_arg in input_args]) == 0:
        input_args.append("--configs_yml=configs/image_wh.yml")
    with ConfigContext(parse_args(input_args)):
        print(args().configs_yml)
        processor = Image_processor()
        inputs = args().inputs
        print('inputs', inputs)
        if not os.path.exists(inputs):
            print("Didn't find the target directory: {}. \n Running the code on the demo images".format(inputs))
            exit()

        if hasattr(args(), 'ground_cam_root'):
            ground_cam_root=args().ground_cam_root
        else:
            print('lack ground_cam_root')
            exit()
        if hasattr(args(), 'scene_type'):
            scene_type=args().scene_type
        else:
            print('lack scene_type')
            exit()
        processor.run(inputs, ground_cam_root, scene_type, args().test_dataset)


if __name__ == '__main__':
    main()