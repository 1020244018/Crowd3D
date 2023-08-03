import os, cv2, yaml
import numpy as np
from lib.utils import write_pkl, read_pkl

def joints2bbox(person_kps):  # kps bbox
    w1 = min(person_kps[:, 0])
    w2 = max(person_kps[:, 0])
    h1 = min(person_kps[:, 1])
    h2 = max(person_kps[:, 1])
    return (w1, h1, w2, h2)


def compute_iou_(pos1, pos2):  # iou for kps
    # print(pos1, pos2)
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    area_sum = area1 + area2
    # print(area1, area2)

    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / area2  # (area_sum - inter)

def visual_kps_limbs_single(image, kps, color_type='kps', padding_size=None):
    kps = kps.copy()
    if padding_size is not None:
        padding_w, padding_h = padding_size
        kps[:, 0] += padding_w
        kps[:, 1] += padding_h

    format = 'coco'
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    part_line = {}

    # draw kps
    color = np.array(np.random.rand(3)) * 255
    per_kps = kps[:, :2]
    kp_scores = kps[:, 2]
    circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.05) + 1
    for i, coord in enumerate(per_kps):
        x_coord, y_coord = int(coord[0]), int(coord[1])
        part_line[i] = (x_coord, y_coord)
        if color_type == 'kps':
            color = p_color[i]
        cv2.circle(image, (x_coord, y_coord), circle_size, color, -1)

    # draw limb
    limb_size = circle_size
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            if color_type == 'kps':
                color = p_color[i]
            if i < len(line_color):
                cv2.line(image, start_xy, end_xy, color, limb_size)

    return image

def innerKpsState(bbox, kps):  # is each point in the kps inner the bbox? yes:1 no:0
    state = np.zeros(kps.shape[0])
    w1, h1, w2, h2 = bbox
    for i in range(kps.shape[0]):
        w_coord, h_coord = int(kps[i][0]), int(kps[i][1])
        if w_coord >= w1 and w_coord <= w2 and h_coord >= h1 and h_coord <= h2:  # inner
            state[i] = 1
    return state

def generate_test_train(mid_root, scene_image_name, test_and_visual=False):
    name_pre=scene_image_name.replace('.jpg', '')
    save_root=mid_root # os.path.join(mid_root, 'testset_annots')
    os.makedirs(save_root, exist_ok=True)
    save_pkl_name=os.path.join(save_root, 'annots.pkl')
    test_base_path=os.path.join(save_root, 'test_train_check')
    
    
    kps_tre=0.4
    joint_num_tre=5

    patch2kps_list=[]
    statistic_dict={}

    joints_path=os.path.join(mid_root,'joints_2d.npy') # joints_2d_tune has two many error joints
    position_path=os.path.join(mid_root, 'position.npy')
    row_col_path=os.path.join(mid_root, 'row_col.npy')
    part_base_path=os.path.join(mid_root, 'part_images')
    

    all_joints=np.load(joints_path)
    position = np.load(position_path)
    row_col = np.load(row_col_path)
    if name_pre not in statistic_dict:
        statistic_dict[name_pre]=0
    
    for i in range(position.shape[0]):
            h1, w1, h2, w2 = position[i].astype(int)
            row, col = row_col[i].astype(int)
            cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(h1) + '_' + str(w1) + '_' + str(
                h2) + '_' + str(
                w2) + '.jpg'
            patch_as_bbox=(w1, h1, w2, h2)
            cur_patch_kps_list=[]
            overlap_num=0
            for kps in all_joints: 
                kps_bbox = joints2bbox(kps)
                iou_for_kps = compute_iou_(patch_as_bbox, kps_bbox)
                if iou_for_kps > kps_tre:  # inner or overlap
                    if iou_for_kps == 1:  # inner
                        pass
                    else:  # overlap
                        temp_state = innerKpsState(patch_as_bbox, kps)
                        if temp_state.sum() < joint_num_tre:
                            continue
                        overlap_num+=1

                    copy_kps = kps.copy()
                    copy_kps[:, 0] -= w1  # to local pixel position
                    copy_kps[:, 1] -= h1
                    cur_patch_kps_list.append(copy_kps)
                    
            if len(cur_patch_kps_list) > overlap_num: # must exist full body
                patch2kps_list.append([cur_name, cur_patch_kps_list])   
                statistic_dict[name_pre]+=1
                
                        

    write_pkl(save_pkl_name, patch2kps_list)
    print('num of image', len(patch2kps_list))
    print(statistic_dict)

    # test and visual
    visual_num = 10
    padding = True
    padding_ratio = 0.3
    if test_and_visual:
        result_list = read_pkl(save_pkl_name)
        np.random.seed(1)
        check_inds = (np.random.choice(np.arange(len(result_list) - 1), size=visual_num, replace=False)).astype(
            np.int64)
        # print(check_inds)
        os.makedirs(test_base_path, exist_ok=True)
        for ind in range(len(result_list)): #check_inds:
            patch_path, cur_patch_kps_list = result_list[ind]
            name_pre=patch_path.split('_')[0]+'_'+patch_path.split('_')[1]
            part_base_path=os.path.join(mid_root, 'part_images')
            patch_image = cv2.imread(os.path.join(part_base_path, patch_path))

            if padding:
                h, w, _ = patch_image.shape
                padding_w = int(w * padding_ratio)
                padding_h = int(h * padding_ratio)
                new_h = h + 2 * padding_h
                new_w = w + 2 * padding_w
                patch_image_padded = np.ones((new_h, new_w, 3)).astype(patch_image.dtype) * 255
                patch_image_padded[padding_h:padding_h + h, padding_w:padding_w + w, :] = patch_image

                for kps in cur_patch_kps_list:
                    patch_image_padded = visual_kps_limbs_single(patch_image_padded, kps, color_type='kps',
                                                                    padding_size=(padding_w, padding_h))

                cv2.imwrite(os.path.join(test_base_path, patch_path), patch_image_padded)
            else:
                for kps in cur_patch_kps_list:
                    patch_image = visual_kps_limbs_single(patch_image, kps, color_type='kps')
                cv2.imwrite(os.path.join(test_base_path, patch_path), patch_image)


def generate_yml(save_name, base, add_info={}):
    with open(base, 'r') as rf:
        info=yaml.load(rf, Loader=yaml.FullLoader)
        
    if 'ARGS' in add_info:
        for key in add_info['ARGS']:
            info['ARGS'][key]=add_info['ARGS'][key]

    with open(save_name, 'w', encoding='utf-8') as wf:
        yaml.dump(info, wf)




def generate_test_train_scale(mid_root, scene_image_name, save_images=False, test_and_visual=False):
    name_pre=scene_image_name.replace('.jpg', '')
    save_root=mid_root # os.path.join(mid_root, 'testset_annots')
    os.makedirs(save_root, exist_ok=True)
    save_pkl_name=os.path.join(save_root, 'optim_annots_scale.pkl')
    test_base_path=os.path.join(save_root, 'test_train_check_scale')
    save_scale_image_root=os.path.join(save_root, 'test_train_images_scale')
    if save_images:
        os.makedirs(save_scale_image_root, exist_ok=True)

    kps_tre=0.4
    joint_num_tre=5

    patch2kps_list=[]
    statistic_dict={}

    joints_path=os.path.join(mid_root,'joints_2d_alphapose_merge_filter.npy') # joints_2d_tune has two many error joints
    position_path=os.path.join(mid_root, 'position.npy')
    row_col_path=os.path.join(mid_root, 'row_col.npy')
    part_base_path=os.path.join(mid_root, 'part_images_scale')
    if not os.path.exists(part_base_path):
        part_base_path=os.path.join(mid_root, 'part_images')
        if not os.path.exists(part_base_path):
            print('Can not find the folder part_images_scale or part_images')
            exit()

    all_joints=np.load(joints_path)
    position = np.load(position_path)
    row_col = np.load(row_col_path)
    if name_pre not in statistic_dict:
        statistic_dict[name_pre]=0
    
    for i in range(position.shape[0]):
            h1, w1, h2, w2 = position[i].astype(int)
            row, col = row_col[i].astype(int)
            cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(h1) + '_' + str(w1) + '_' + str(
                h2) + '_' + str(
                w2) + '.jpg'
            patch_as_bbox=(w1, h1, w2, h2)
            cur_size=w2-w1
            cur_patch_kps_list=[]
            cur_patch_kps_scale_list=[]
            overlap_num=0
            for kps in all_joints: 
                kps_bbox = joints2bbox(kps)
                iou_for_kps = compute_iou_(patch_as_bbox, kps_bbox)
                if iou_for_kps > kps_tre:  # inner or overlap
                    if iou_for_kps == 1:  # inner
                        pass
                    else:  # overlap
                        temp_state = innerKpsState(patch_as_bbox, kps)
                        if temp_state.sum() < joint_num_tre:
                            continue
                        overlap_num+=1

                    copy_kps = kps.copy()
                    copy_kps[:, 0] -= w1  # to local pixel position
                    copy_kps[:, 1] -= h1
                    cur_patch_kps_list.append(copy_kps)
                    cur_patch_kps_scale_list.append(copy_kps / cur_size * 512)
                    
            if len(cur_patch_kps_list) > overlap_num: # must exist full body
                # patch2kps_list.append([cur_name, cur_patch_kps_list])   
                patch2kps_list.append([cur_name, cur_patch_kps_scale_list]) 
                statistic_dict[name_pre]+=1


    write_pkl(save_pkl_name, patch2kps_list)
    # print('num of image', len(patch2kps_list))
    # print(statistic_dict)

    # if save_images:
    #     # generate_scale_images_for_test_train
    #     for i in  range(len(patch2kps_list)):
    #         patch_name, _ = patch2kps_list[i]
    #         patch_image = cv2.imread(os.path.join(part_base_path, patch_name))
    #         patch_image = cv2.resize(patch_image, (512, 512))
    #         cv2.imwrite(os.path.join(save_scale_image_root, patch_name), patch_image)

    # test and visual
    visual_num = 10
    padding = True
    padding_ratio = 0.3
    if test_and_visual:
        result_list = read_pkl(save_pkl_name)
        np.random.seed(1)
        check_inds = (np.random.choice(np.arange(len(result_list) - 1), size=visual_num, replace=False)).astype(
            np.int64)
        # print(check_inds)
        os.makedirs(test_base_path, exist_ok=True)
        for ind in check_inds:
            patch_path, cur_patch_kps_list = result_list[ind]
            name_pre=patch_path.split('_')[0]+'_'+patch_path.split('_')[1]
            # part_base_path=os.path.join(mid_root, 'part_images_scale')
            patch_image = cv2.imread(os.path.join(part_base_path, patch_path))
            if patch_image.shape[0]!=512:
                patch_image=cv2.resize(patch_image, (512, 512))

            if padding:
                h, w, _ = patch_image.shape
                padding_w = int(w * padding_ratio)
                padding_h = int(h * padding_ratio)
                new_h = h + 2 * padding_h
                new_w = w + 2 * padding_w
                patch_image_padded = np.ones((new_h, new_w, 3)).astype(patch_image.dtype) * 255
                patch_image_padded[padding_h:padding_h + h, padding_w:padding_w + w, :] = patch_image

                for kps in cur_patch_kps_list:
                    patch_image_padded = visual_kps_limbs_single(patch_image_padded, kps, color_type='kps',
                                                                    padding_size=(padding_w, padding_h))

                cv2.imwrite(os.path.join(test_base_path, patch_path), patch_image_padded)
            else:
                for kps in cur_patch_kps_list:
                    patch_image = visual_kps_limbs_single(patch_image, kps, color_type='kps')
                cv2.imwrite(os.path.join(test_base_path, patch_path), patch_image)
        
    

def generate_test_train_scale_for_scene(mid_root, save_root, joints_file, test_and_visual=False):
    
    frame_list=os.listdir(mid_root)
    os.makedirs(save_root, exist_ok=True)
    save_pkl_name=os.path.join(save_root, 'annots_scale.pkl')
    visual_folder=os.path.join(save_root, 'test_train_check_scale')
    save_scale_image_root=os.path.join(save_root, 'test_train_images_scale')
    os.makedirs(save_scale_image_root, exist_ok=True)

    kps_tre=0.4
    joint_num_tre=5
    
    all_patch2kps_list=[]
    all_statistic_dict={}
    
    for frame_name in frame_list:
        frame_folder=os.path.join(mid_root, frame_name)
        name_pre=frame_name
        
        joints_path=os.path.join(frame_folder, joints_file) # joints_2d_tune has too many error joints
        position_path=os.path.join(frame_folder, 'position.npy')
        row_col_path=os.path.join(frame_folder, 'row_col.npy')
        part_base_path=os.path.join(frame_folder, 'part_images')
        part_scale_base_path=os.path.join(frame_folder, 'part_images_scale')

        all_joints=np.load(joints_path)
        position = np.load(position_path)
        row_col = np.load(row_col_path)
        
        patch2kps_list=[]
        statistic_dict={}
    
        if name_pre not in statistic_dict:
            statistic_dict[name_pre]=0
        
        for i in range(position.shape[0]):
                h1, w1, h2, w2 = position[i].astype(int)
                row, col = row_col[i].astype(int)
                cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(h1) + '_' + str(w1) + '_' + str(
                    h2) + '_' + str(
                    w2) + '.jpg'
                patch_as_bbox=(w1, h1, w2, h2)
                cur_size=w2-w1
                cur_patch_kps_list=[]
                cur_patch_kps_scale_list=[]
                overlap_num=0
                for kps in all_joints: 
                    kps_bbox = joints2bbox(kps)
                    iou_for_kps = compute_iou_(patch_as_bbox, kps_bbox)
                    if iou_for_kps > kps_tre:  # inner or overlap
                        if iou_for_kps == 1:  # inner
                            pass
                        else:  # overlap
                            temp_state = innerKpsState(patch_as_bbox, kps)
                            if temp_state.sum() < joint_num_tre:
                                continue
                            overlap_num+=1

                        copy_kps = kps.copy()
                        copy_kps[:, 0] -= w1  # to local pixel position
                        copy_kps[:, 1] -= h1
                        cur_patch_kps_list.append(copy_kps)
                        cur_patch_kps_scale_list.append(copy_kps / cur_size * 512)
                        
                if len(cur_patch_kps_list) > overlap_num: # must exist full body
                    # patch2kps_list.append([cur_name, cur_patch_kps_list])   
                    patch2kps_list.append([cur_name, cur_patch_kps_scale_list]) 
                    statistic_dict[name_pre]+=1
        
        # generate_scale_images_for_test_train
        for i in  range(len(patch2kps_list)):
            patch_name, _ = patch2kps_list[i]
            
            # patch_image = cv2.imread(os.path.join(part_base_path, patch_name))
            # patch_image = cv2.resize(patch_image, (512, 512))
            # cv2.imwrite(os.path.join(save_scale_image_root, patch_name), patch_image)
            patch_path=os.path.join(part_scale_base_path, patch_name)
            command='cp '+patch_path + ' '+save_scale_image_root
            os.system(command)

        all_patch2kps_list=all_patch2kps_list+ patch2kps_list
        all_statistic_dict.update(statistic_dict)

    write_pkl(save_pkl_name, all_patch2kps_list)
    print('total num of image', len(all_patch2kps_list))
    print(all_statistic_dict)

        
    # test and visual
    visual_num = 10
    padding = True
    padding_ratio = 0.3
    if test_and_visual:
        result_list = read_pkl(save_pkl_name)
        np.random.seed(1)
        check_inds = (np.random.choice(np.arange(len(result_list) - 1), size=visual_num, replace=False)).astype(
            np.int64)
        # print(check_inds)
        os.makedirs(visual_folder, exist_ok=True)
        for ind in check_inds:
            patch_path, cur_patch_kps_list = result_list[ind]
            name_pre=patch_path.split('_')[0]+'_'+patch_path.split('_')[1]
 
            patch_image = cv2.imread(os.path.join(save_scale_image_root, patch_path))
            if patch_image.shape[0]!=512:
                patch_image=cv2.resize(patch_image, (512, 512))

            if padding:
                h, w, _ = patch_image.shape
                padding_w = int(w * padding_ratio)
                padding_h = int(h * padding_ratio)
                new_h = h + 2 * padding_h
                new_w = w + 2 * padding_w
                patch_image_padded = np.ones((new_h, new_w, 3)).astype(patch_image.dtype) * 255
                patch_image_padded[padding_h:padding_h + h, padding_w:padding_w + w, :] = patch_image

                for kps in cur_patch_kps_list:
                    patch_image_padded = visual_kps_limbs_single(patch_image_padded, kps, color_type='kps',
                                                                    padding_size=(padding_w, padding_h))

                cv2.imwrite(os.path.join(visual_folder, patch_path), patch_image_padded)
            else:
                for kps in cur_patch_kps_list:
                    patch_image = visual_kps_limbs_single(patch_image, kps, color_type='kps')
                cv2.imwrite(os.path.join(visual_folder, patch_path), patch_image)