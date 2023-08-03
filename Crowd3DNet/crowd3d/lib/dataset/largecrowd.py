import sys, os

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from dataset.image_base import *


class LargeCrowd(Image_base):
    def __init__(self, train_flag=True,
                 regress_smpl=True, **kwargs):
        super(LargeCrowd, self).__init__(train_flag, regress_smpl)
        self.data_folder = os.path.join(args().dataset_rootdir, 'largecrowd_scale')
        # select_full_scale_patch_To_kps_fbbox_list_hvip_5scene.pkl
        self.patch_path= os.path.join(self.data_folder, 'annots_scale.pkl')
        self.image_root=os.path.join(self.data_folder, 'images')
        self.ground_path=os.path.join(self.data_folder, 'ground_info')
        self.min_pts_required = 2
        self.compress_length = 1.25 # 1.15
        self.init()
        logging.info('Loaded largecrowd data,total {} samples'.format(
            self.__len__()))

    def load_ground_cam(self, root_path):
        ground_dict = {}
        cam_dict = {}
        scene_type_list = os.listdir(root_path)
        for scene_type in scene_type_list:
            ground_dict[scene_type] = np.load(os.path.join(root_path, scene_type, 'ground.npy'))
            cam_dict[scene_type] = np.load(os.path.join(root_path, scene_type, 'cam_para.npy'))
        return ground_dict, cam_dict

    def init(self):
        self.patch_data = load_pkl(self.patch_path)
        self.file_paths = self.patch_data
        self.joint_mapper = constants.joint_mapping(constants.COCO_17, constants.SMPL_ALL_54)
        self.ground, self.cam = self.load_ground_cam(self.ground_path)
        self.scene_shape_dict={
            'playground0_00':[19200, 6515],
            'playground0_01':[19200, 6515],
            'parkingLot_00':[19200, 6271],
            'jinJie_00':[19200, 6359],
            'jinJie_01':[19200, 6337]
        }
        
    def determine_visible_person(self, kp2ds, width, height):
        visible_person_id,kp2d_vis = [],[]
        for person_id,kp2d in enumerate(kp2ds):
            visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height))
            if visible_kps_mask.sum()>1:
                visible_person_id.append(person_id)
                kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
        return np.array(visible_person_id), np.array(kp2d_vis)

    def get_image_info(self, index):
        if self.compress_length>1:
            index = index * self.compress_length + random.randint(0, int(round(self.compress_length)) - 1)
            index = int(round(index))
        patch_path, cur_patch_kps_list, _, _, cur_patch_hvip2d_list= self.patch_data[index % len(self.patch_data)]
        name_info=patch_path.split('_')
        scene_type=name_info[0]+'_'+name_info[1]

        imgpath = os.path.join(self.image_root,  patch_path)
        
        
        image = cv2.imread(imgpath)[:, :, ::-1]
        height, width, _ = image.shape

        kp2ds, bbox, valid_mask_2d, valid_mask_3d, params = [], [], [], [], None
        valid_mask_hvip2d = []
        ankle2ds=[]
        ankle_index=[15, 16] # for coco17
        visible_person_id, cur_patch_kps_list_vis=self.determine_visible_person(cur_patch_kps_list,width, height)


        hvip2ds=[]
        hvip2ds_flag=[]

        for ii in visible_person_id:
            hvip2d=cur_patch_hvip2d_list[ii]
            if hvip2d is None:
                hvip2ds.append([-2, -2])
                hvip2ds_flag.append(False)
            else:
                hvip2ds.append(hvip2d)
                hvip2ds_flag.append(True)
        hvip2ds=np.array(hvip2ds)

        real_tcs=[]
        torso_index=[constants.COCO_17['L_Shoulder'],constants.COCO_17['R_Shoulder'],constants.COCO_17['L_Hip'],constants.COCO_17['R_Hip']]

        for idx in range(len(cur_patch_kps_list_vis)):
            joint=cur_patch_kps_list_vis[idx]
            ankle_2d=joint[ankle_index, :2]
            ankle2ds.append(ankle_2d)
            real_tcs.append(joint[torso_index, :2].mean(0))


            invis_kps = joint[:, -1] < 0.1
            joint[invis_kps] = -2.

            joint = self.map_kps(joint, maps=self.joint_mapper)
            kp2ds.append(joint)
            # bbox.append(cur_patch_bbox_list[idx])  # left, top, right, bottom
            valid_mask_2d.append([True, True, True])
            valid_mask_3d.append(self.default_valid_mask_3d)

            ankle_flag=True
            # if ankle_2d[0,0]==-2 or ankle_2d[1,0]==-2:
            #     ankle_flag=False
            valid_mask_hvip2d.append([hvip2ds_flag[idx], False, ankle_flag, False])


        scene_shape=np.array(self.scene_shape_dict[scene_type])
        ground = np.array(self.ground[scene_type]).reshape(-1)  # (4,)
        ankle2ds=np.array(ankle2ds)
        real_tcs=np.array(real_tcs).reshape(-1, 1, 2)

        camK=np.array(self.cam[scene_type]).reshape(3,3)

        path_info=patch_path.split('_')
        patch_top, patch_left = int(path_info[-4]), int(path_info[-3])
        orign_size=int(path_info[-2]) - int(path_info[-4])
        data_scale=orign_size / args().input_size

        patch_leftTop = np.array([patch_left, patch_top]).astype(int)

        assert hvip2ds.shape[0]== np.array(kp2ds).shape[0], 'error !'

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape

        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None, \
                    'vmask_2d': np.array(valid_mask_2d), 'vmask_3d': np.array(valid_mask_3d), \
                    'kp3ds': None, 'params': params, 'img_size': image.shape[:2], 'ds': 'pandapose', \
                    'ground': ground, 'camK': camK, 'dist': np.zeros(5),
                    'patch_leftTop': patch_leftTop, 'vmask_foot': np.array(valid_mask_hvip2d),
                    'hvip2ds': hvip2ds, 'bbox': None, 'trans': None, 'root_cam': None, 'ankle2ds': ankle2ds,
                    'data_scale':data_scale, 'real_tcs': real_tcs, 'scene_shape':scene_shape}

        return img_info



    def __len__(self):
        return int(len(self.file_paths) // self.compress_length)


def load_pkl(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    dataset = LargeCrowd(train_flag=True, regress_smpl=False)
    print('len of dataset', dataset.__len__())
    sel = list(range(500))
    for i in sel:
        data = dataset.__getitem__(i)



    print('Done')
