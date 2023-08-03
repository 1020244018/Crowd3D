import sys, os
import matplotlib.pyplot as plt
import numpy as np
import time

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from config import args
from dataset.image_base import *
import pickle
from dataset.camera_parameters import project_to_2d, project_to_2d_linear

class AGORA(Image_base):

    def __init__(self, train_flag=True, regress_smpl=True, **kwargs):
        super(AGORA, self).__init__(train_flag, regress_smpl)
        self.data_folder = os.path.join(args().dataset_rootdir,
                                        'agora_scale')
        self.image_folder = os.path.join(self.data_folder, 'images')
        self.annots_file = os.path.join(self.data_folder, 'annots_scale.pkl')

        self.train_flag = train_flag
        if self.regress_smpl:
            self.smplr = SMPLR(use_gender=False)
            self.root_inds = None

        self.joint_mapper = constants.joint_mapping(constants.SMPL_ALL_54,
                                                    constants.SMPL_ALL_54)
        if self.train_flag and self.regress_smpl:
            self.joint3d_mapper_smpl = constants.joint_mapping(
                constants.SMPL_ALL_54, constants.SMPL_ALL_54)
        else:
            self.joint3d_mapper = constants.joint_mapping(
                constants.SMPL_ALL_54, constants.SMPL_ALL_54)

        # self.kps_vis = (self.joint_mapper != -1).astype(np.float32)[:, None]

        self.shuffle_mode = args().shuffle_crop_mode
        self.shuffle_ratio = args().shuffle_crop_ratio_3d

        self.compress_length = 1

        self.load_file_list()


        logging.info('Loaded agora data,total {} samples'.format(
            self.__len__()))

    def load_file_list(self):
        with open(self.annots_file, 'rb') as rf:
            self.annots = pickle.load(rf)
        self.file_paths = list(self.annots.keys())

    def determine_visible_person(self, kp2ds, width, height):
        visible_person_id, kp2d_vis = [], []
        for person_id in range(kp2ds.shape[0]):
            kp2d = kp2ds[person_id]
            visible_kps_mask = np.logical_and(
                np.logical_and(0 < kp2d[:, 0], kp2d[:, 0] < width),
                np.logical_and(0 < kp2d[:, 1], kp2d[:, 1] < height,
                               kp2d[:, 2] > 0))
            if visible_kps_mask.sum() > 1:
                visible_person_id.append(person_id)
                kp2d_vis.append(
                    np.concatenate([kp2d[:, :2], visible_kps_mask[:, None]],
                                   1))
        return np.array(visible_person_id), np.array(kp2d_vis)

    def get_image_info(self, index):
        imgpath = os.path.join(self.image_folder,
                               self.file_paths[index % len(self.file_paths)])
        img_name = os.path.basename(imgpath)
        info = self.annots[img_name].copy()

        image = cv2.imread(imgpath)[:, :, ::-1]  # convert BGR to RGB
        height, width, _ = image.shape

        pose_params = info['pose_params']
        beta_params = info['beta_params']
        ground_cam = info['ground']
        camMatrix = info['camMatrix']

        gt_joints_2d_54 = info['gt_joints_2d_all_vis']
        gt_joints_3d_54 = info['gt_joints_3d_all']


        hvip2d = info['foot2d_torso_center']
        trans = info['trans3d_angle_mean_and_foot3d']  # N,3
        kids_flag= info['kids_flag'] # N,
        

        valid_mask_2d, valid_mask_3d, params = [], [], None
        visible_person_id, gt_joints_2d_54_vis = self.determine_visible_person(
            gt_joints_2d_54, width, height)
        
        gt_joints_3d_vis = gt_joints_3d_54[visible_person_id] # m, 54, 3
        hvip2d = hvip2d[visible_person_id]
        trans = trans[visible_person_id]
        kids_flag=kids_flag[visible_person_id]
        
        hip_index=[45, 46]
        hips=gt_joints_3d_vis[:, hip_index,:]
        root_cam=hips.mean(1)


        if self.regress_smpl:
            pose = pose_params.reshape(-1, 72)
            beta = beta_params.reshape(-1, 10)
            smpl_outs = self.smplr(pose, beta)
            kp3ds = smpl_outs['j3d'].numpy()[visible_person_id]
            for i in range(kp3ds.shape[0]):
                if kids_flag[i]:  # kid, no supervise
                    valid_mask_3d.append([False, False, False, False])
                else:
                    valid_mask_3d.append([True, True, True, True])

            # print('smpl_j3d', kp3ds[0] - kp3ds[0][0])

        else:
            kp3ds = []
            for joint_3d in gt_joints_3d_vis:
                joint = self.map_kps(joint_3d, maps=self.joint3d_mapper)
                kp3ds.append(joint)
                valid_mask_3d.append([True, False, False, False])
            kp3ds = np.array(kp3ds)
        # print('gt_joints_3d_vis', gt_joints_3d_vis[0]-gt_joints_3d_vis[0][0])
        kp2ds = []
        valid_mask_hvip2d=[]
        ankle_index=[7, 8] # for agora
        ankle2ds =[]
        real_tcs=[]
        torso_index=[constants.SMPL_ALL_54['L_Shoulder'], constants.SMPL_ALL_54['R_Shoulder'],constants.SMPL_ALL_54['R_Hip'],constants.SMPL_ALL_54['L_Hip']]
        for joint_2d in gt_joints_2d_54_vis:
            real_tcs.append(joint_2d[torso_index, :2].mean(0))
            invis_kps = joint_2d[:, -1] < 0.1
            ankle_2d=joint_2d[ankle_index, :2]
            ankle2ds.append(ankle_2d)

            joint_2d[invis_kps] = -2.

            joint = self.map_kps(joint_2d, maps=self.joint_mapper)
            # joint = np.concatenate([joint, self.kps_vis], 1)
            kp2ds.append(joint)
            valid_mask_2d.append([True, True, True])
            valid_mask_hvip2d.append([True, True, True, True])


        kp2ds = np.array(kp2ds)
        ankle2ds=np.array(ankle2ds) # N, 2, 2
        real_tcs=np.array(real_tcs).reshape(-1, 1, 2)

        if self.regress_smpl:
            params = np.concatenate([pose, beta], axis=-1)

        assert kp2ds.shape[0] == kp3ds.shape[0] == hvip2d.shape[0] == trans.shape[
            0], 'error: kp2ds.shape[0]==kp3ds.shape[0]==hvip2d.shape[0]==trans.shape[0]'
        

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        # vmask_foot | 0: hvip2d | 1: trans3d | 2: ankle | 3: root_cam
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None, \
                    'vmask_2d': np.array(valid_mask_2d), 'vmask_3d': np.array(valid_mask_3d), \
                    'kp3ds': kp3ds, 'params': params, 'img_size': image.shape[:2], 'ds': 'agora', \
                    'ground': ground_cam, 'camK': camMatrix, 'dist': np.zeros(5),
                    'patch_leftTop': np.array([0, 0]).astype(int), 'vmask_foot': np.array(valid_mask_hvip2d),
                    'hvip2ds': hvip2d, 'bbox': None, 'trans': trans, 'root_cam': root_cam, 'ankle2ds': ankle2ds,
                    'data_scale': 1280/args().input_size, 'real_tcs': real_tcs, 
                    'scene_shape': np.array([1280, 720])}
            

        return img_info

    def __len__(self):
        if self.train_flag:
            return len(self.file_paths) // self.compress_length
        else:
            return len(self.file_paths)

    def project_point(self, joint, RT, KKK):
        P = np.dot(KKK, RT)
        joints_2d = np.dot(P, joint)
        joints_2d = joints_2d[0:2] / joints_2d[2]

        return joints_2d


if __name__ == '__main__':
    # args().configs_yml = 'cui.yml'
    # args().model_version = 7
    dataset = AGORA(train_flag=True, regress_smpl=True)
    print('len of dataset', dataset.__len__())
    # sel = range(dataset.__len__())
    sel =[i for i in range(dataset.__len__())]
    all_delta2d=[]
    for i in sel:
        data = dataset.__getitem__(i)
  

        # for key, value in data.items():
        #     if isinstance(value, str):
        #         print(key, value)
        #     else:
        #         print(key, value.shape)
        # exit()

    # generate for test
    generate_flag = False
    num = 10
    save_root = 'generate_select/agora_select'
    save_image_path = os.path.join(save_root, 'images')
    save_input_params_name_pkl = 'input_params.pkl'
    os.makedirs(save_image_path, exist_ok=True)

    total_lens=dataset.__len__()
    select_index=np.random.permutation(total_lens)[:num]

    if generate_flag:
        input_params = {}
        for i in range(num):
            info, image_input = dataset.get_content_for_test(select_index[i])
            image_path = '\'' + info['imgpath'] + '\''
            os.system('cp ' + image_path + ' ' + save_image_path)

            key = os.path.basename(info['imgpath'])
            input_params[key] = {'ground': info['ground']}
            input_params[key]['camK'] = info['camK']
            input_params[key]['dist'] = info['dist']
            input_params[key]['patch_leftTop'] = info['patch_leftTop']

        import pickle
        with open(os.path.join(save_root, save_input_params_name_pkl),
                  'wb') as wf:
            pickle.dump(input_params, wf)
        print('save as', os.path.join(save_root, save_input_params_name_pkl))

    print('Done')
