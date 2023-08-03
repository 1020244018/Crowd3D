import sys, os
from unittest.mock import _patch_dict

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from dataset.image_base import *

class TEST_TRAIN_SCALE(Image_base):
    def __init__(self, train_flag=True,
                 regress_smpl=True, **kwargs):
        super(TEST_TRAIN_SCALE, self).__init__(train_flag, regress_smpl)

        self.anno_path=args().test_train_anno_path
        self.part_image_base_path=args().test_train_image_path
        self.ground_path=args().test_train_ground_path
        self.min_pts_required = 2
        self.compress_length = 1
        self.init_test()
        logging.info('Panda 2D keypoint and bbox data has been loaded, total {} samples'.format(len(self)))

    def load_ground_cam(self, root_path):
        ground_dict = {}
        cam_dict = {}
        scene_type_list = os.listdir(root_path)
        for scene_type in scene_type_list:
            ground_dict[scene_type] = np.load(os.path.join(root_path, scene_type, 'ground.npy'))
            cam_dict[scene_type] = np.load(os.path.join(root_path, scene_type, 'cam_para.npy'))
        return ground_dict, cam_dict

    def init_test(self, ):
        self.patch_data = load_pkl(self.anno_path)
        self.file_paths = self.patch_data
        self.joint_mapper = constants.joint_mapping(constants.COCO_17, constants.SMPL_ALL_54)
        self.ground=np.load(os.path.join(self.ground_path, 'ground.npy'))
        self.cam=np.load(os.path.join(self.ground_path, 'cam_para.npy'))
        self.scene_shape=np.load(os.path.join(self.ground_path, 'scene_shape.npy'))


    def determine_visible_person(self, kp2ds, width, height):
        visible_person_id,kp2d_vis = [],[]
        for person_id,kp2d in enumerate(kp2ds):
            visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height))
            if visible_kps_mask.sum()>1:
                visible_person_id.append(person_id)
                kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
        return np.array(visible_person_id), np.array(kp2d_vis)

    def get_image_info(self, index):
        # index = index * self.compress_length + random.randint(0, self.compress_length - 1)
        patch_path, cur_patch_kps_list = self.patch_data[index % len(self.patch_data)]

        imgpath = os.path.join(self.part_image_base_path, patch_path)
        image = cv2.imread(imgpath)[:, :, ::-1]

        height, width, _ = image.shape
        kp2ds, bbox, valid_mask_2d, valid_mask_3d, params = [], [], [], [], None
        valid_mask_hvip2d = []
        ankle_index=[15, 16]
        ankle2ds=[]
        visible_person_id, cur_patch_kps_list_vis=self.determine_visible_person(cur_patch_kps_list,width, height)
        for idx in range(len(cur_patch_kps_list_vis)):
            joint=cur_patch_kps_list_vis[idx]
            ankle_2d=joint[ankle_index, :2]
            ankle2ds.append(ankle_2d)

            invis_kps = joint[:, -1] < 0.1
            joint[invis_kps] = -2.

            joint = self.map_kps(joint, maps=self.joint_mapper)
            kp2ds.append(joint)
            # bbox.append(cur_patch_bbox_list[idx])  # left, top, right, bottom
            valid_mask_2d.append([True, True, True])
            valid_mask_3d.append(self.default_valid_mask_3d)

            ankle_flag=True

            valid_mask_hvip2d.append([False, False, ankle_flag, False])


        ground = np.array(self.ground).reshape(-1)  # (4,)

        ankle2ds=np.array(ankle2ds)

        camK=np.array(self.cam).reshape(3,3)
        infos=patch_path.split('_')
        h1, h2 = int(infos[-4]), int(infos[-2])
        w1, w2 = int(infos[-3]), int(infos[-1][:-4])
        cur_size = h2 - h1
        

        patch_top, patch_left = h1, w1
        patch_leftTop = np.array([patch_left, patch_top]).astype(int)

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape

        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None, \
                    'vmask_2d': np.array(valid_mask_2d), 'vmask_3d': np.array(valid_mask_3d), \
                    'kp3ds': None, 'params': params, 'img_size': image.shape[:2], 'ds': 'pandapose', \
                    'ground': ground, 'camK': camK, 'dist': np.zeros(5),
                    'patch_leftTop': patch_leftTop, 'vmask_foot': np.array(valid_mask_hvip2d),
                    'hvip2ds': None, 'bbox': None, 'trans': None, 'root_cam': None,'ankle2ds': ankle2ds,
                    'data_scale':cur_size/512, 'scene_shape': self.scene_shape}

        return img_info

    def __len__(self):
        return len(self.file_paths) // self.compress_length


def load_pkl(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    dataset = TEST_TRAIN(train_flag=True, regress_smpl=False)
    print('len of dataset', dataset.__len__())
    sel = [1, 2, 3, 4, 5]
    for i in sel:
        data = dataset.__getitem__(i)
    # for key, value in data.items():
    #     if isinstance(value, str):
    #         print(key, value)
    #     else:
    #         print(key, value.shape)

    # generate for test

    print('Done')
