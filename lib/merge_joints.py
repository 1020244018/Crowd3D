from curses import curs_set
import json
import os
import numpy as np
import cv2
import json
import copy
import queue

def get_candicate_tre(all_kps_bbox, ratio=0.8):
    h_list=[kps_bbox[3] - kps_bbox[1] for kps_bbox in all_kps_bbox]
    h_list_sorted=sorted(h_list)
    index=int(len(h_list_sorted)*ratio)
    print(len(h_list_sorted), h_list_sorted[index])
    return h_list_sorted[index]

def compute_iou(pos1, pos2):  # iou for kps
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
        return inter / (area_sum - inter)

def is_h_close(pos1, pos2, expand_ratio=0.5):  # iou for kps
    # print(pos1, pos2)
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    top1-=(down1-top1)*expand_ratio
    down1+=(down1-top1)*expand_ratio

    if down1 < top2  or top1 > down2:
        return 0
    else:
        return 1
  

def kps2bbox(kps):
    w1=int(kps[:, 0].min())
    w2=int(kps[:, 0].max())
    h1=int(kps[:, 1].min())
    h2=int(kps[:, 1].max())
    return [w1,h1,w2,h2]

def getPosition(name):
    name_list=name.split('_')
    h1=int(name_list[-4])
    h2=int(name_list[-2])
    w1=int(name_list[-3])
    w2=int(name_list[-1].replace('.jpg', ''))
    return (w1,h1,w2,h2)

def analysis_person(json_path, image, visual_name, filter=True, tre=0.2, minmin_h=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    joints = []
    H, W, _ = image.shape
    min_h_mean=99999999
    max_h_mean=-1
    min_h, max_h=0, 0
    for person_dict in data:
        person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
        confident = np.mean(person_kps[:, 2])
        if filter and confident < tre:
            continue
        patch_name=person_dict['image_id']
        w1,h1,w2,h2=getPosition(patch_name)
        person_kps[:, 0]+=w1
        person_kps[:, 1]+=h1
        
        color=np.random.random(3)*255
        kps_bbox=kps2bbox(person_kps)
        width=max([int((kps_bbox[3]-kps_bbox[1])*0.01), 5])
        cv2.rectangle(image, (kps_bbox[0], kps_bbox[1]), (kps_bbox[2], kps_bbox[3]), color=color, thickness=width)

        joints.append(person_kps)
        h_mean=np.mean([kps2bbox[1], kps2bbox[3]])
        if h_mean < min_h_mean:
            temp=max( [int(person_kps[:, 1].max() - person_kps[:, 1].min()), 0])
            if minmin_h > temp:
                continue
            min_h=temp
            min_h_mean=max([h_mean,0])

        if h_mean > max_h_mean:
            max_h_mean=h_mean
            max_h=int(person_kps[:, 1].max() - person_kps[:, 1].min())

    cv2.imwrite(visual_name, image)
    return min_h, max_h, min_h_mean, max_h_mean

def analysis_person_v2(json_path, image, visual_name, filter=True, tre=0.2, minmin_h=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    joints = []
    H, W, _ = image.shape
    min_h_mean=99999999
    max_h_mean=-1
    min_h, max_h=0, 0
    min_kps_bbox=max_kps_bbox=[]

    # all_kps_bbox=[]
    # for person_dict in data:
    #     person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
    #     confident = np.mean(person_kps[:, 2])
    #     if filter and confident < tre:
    #         continue
    #     patch_name=person_dict['image_id']
    #     w1,h1,w2,h2=getPosition(patch_name)
    #     person_kps[:, 0]+=w1
    #     person_kps[:, 1]+=h1
    #     all_kps_bbox.append(kps2bbox(person_kps))
    all_kps_bbox=[]
    for person_dict in data:
        person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
        confident = np.mean(person_kps[:, 2])
        if filter and confident < tre:
            continue
        patch_name=person_dict['image_id']
        w1,h1,w2,h2=getPosition(patch_name)
        person_kps[:, 0]+=w1
        person_kps[:, 1]+=h1
        
        color=np.random.random(3)*255
        kps_bbox=kps2bbox(person_kps)
        all_kps_bbox.append(kps_bbox)
        width=max([int((kps_bbox[3]-kps_bbox[1])*0.01), 5])
        cv2.rectangle(image, (kps_bbox[0], kps_bbox[1]), (kps_bbox[2], kps_bbox[3]), color=color, thickness=width)

        joints.append(person_kps)
        h_mean=(kps_bbox[1]+kps_bbox[3])/2
        if h_mean < min_h_mean:
            temp=max( [int(person_kps[:, 1].max() - person_kps[:, 1].min()), 0])
            if minmin_h > temp:
                continue
            min_h=temp
            min_h_mean=max([h_mean,0])
            min_kps_bbox=kps_bbox

        if h_mean > max_h_mean:
            max_h_mean=h_mean
            max_h=int(person_kps[:, 1].max() - person_kps[:, 1].min())
            max_kps_bbox=kps_bbox
    
    for i in range(len(all_kps_bbox)):
        # for min_kps
        final_min_h=min_h
        final_max_h=max_h
        if compute_iou(min_kps_bbox, all_kps_bbox[i])>0:
            if (all_kps_bbox[i][3] - all_kps_bbox[i][1]) > final_min_h:
                final_min_h=all_kps_bbox[i][3] - all_kps_bbox[i][1]
        if compute_iou(max_kps_bbox, all_kps_bbox[i])>0:
            if (all_kps_bbox[i][3] - all_kps_bbox[i][1]) > final_max_h:
                final_max_h=all_kps_bbox[i][3] - all_kps_bbox[i][1]

    cv2.imwrite(visual_name, image)
    return final_min_h, final_max_h, min_h_mean, max_h_mean

def analysis_person_v3(json_path, image, visual_name, filter=True, tre=0.2, minmin_h=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    joints = []
    H, W, _ = image.shape
    min_h_mean=99999999
    max_h_mean=-1
    min_h, max_h=0, 0
    min_kps_bbox=max_kps_bbox=[]

    # all_kps_bbox=[]
    # for person_dict in data:
    #     person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
    #     confident = np.mean(person_kps[:, 2])
    #     if filter and confident < tre:
    #         continue
    #     patch_name=person_dict['image_id']
    #     w1,h1,w2,h2=getPosition(patch_name)
    #     person_kps[:, 0]+=w1
    #     person_kps[:, 1]+=h1
    #     all_kps_bbox.append(kps2bbox(person_kps))
    all_kps_bbox=[]
    max_h_TopN=10
    max_h_TopN_queue=queue.PriorityQueue()
    pid=0
    for person_dict in data:
        person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
        confident = np.mean(person_kps[:, 2])
        if filter and confident < tre:
            continue
        patch_name=person_dict['image_id']
        w1,h1,w2,h2=getPosition(patch_name)
        person_kps[:, 0]+=w1
        person_kps[:, 1]+=h1
        
        color=np.random.random(3)*255
        kps_bbox=kps2bbox(person_kps)

        width=max([int((kps_bbox[3]-kps_bbox[1])*0.01), 5])
        cv2.rectangle(image, (kps_bbox[0], kps_bbox[1]), (kps_bbox[2], kps_bbox[3]), color=color, thickness=width)

        joints.append(person_kps)
        h_mean=(kps_bbox[1]+kps_bbox[3])/2
        cur_h=max( [int(person_kps[:, 1].max() - person_kps[:, 1].min()), 0])
        if minmin_h > cur_h:
            continue
        
        all_kps_bbox.append(kps_bbox)

        if h_mean < min_h_mean:
            min_h=cur_h
            min_h_mean=max([h_mean,0])
            min_kps_bbox=kps_bbox

        if h_mean > max_h_mean:
            max_h_mean=h_mean
            max_h=cur_h
            max_kps_bbox=kps_bbox
        
        if max_h_TopN_queue.qsize() < max_h_TopN:
            max_h_TopN_queue.put([cur_h, pid])
        else:
            exist_max_h, exist_max_h_pid=max_h_TopN_queue.get()
            if cur_h>exist_max_h:
                max_h_TopN_queue.put([cur_h, pid])
            else:
                max_h_TopN_queue.put([exist_max_h, exist_max_h_pid])
        pid+=1
        
    
    for i in range(len(all_kps_bbox)):
        # for min_kps
        final_min_h=min_h
        if is_h_close(min_kps_bbox, all_kps_bbox[i])>0:
            if (all_kps_bbox[i][3] - all_kps_bbox[i][1]) > final_min_h:
                final_min_h=all_kps_bbox[i][3] - all_kps_bbox[i][1]
        final_max_h=max_h
        if is_h_close(max_kps_bbox, all_kps_bbox[i])>0:
            if (all_kps_bbox[i][3] - all_kps_bbox[i][1]) > final_max_h:
                final_max_h=all_kps_bbox[i][3] - all_kps_bbox[i][1]

    # final_max_h=-1
    # new_h_mean=-1
    # while not max_h_TopN_queue.empty():
    #     cur_h, pid = max_h_TopN_queue.get()
    #     h_mean=all_kps_bbox[pid][3] - all_kps_bbox[pid][1]
    #     if h_mean > new_h_mean:
    #         new_h_mean=h_mean
    #         final_max_h=cur_h
    # max_h_mean=new_h_mean

    cv2.imwrite(visual_name, image)
    return final_min_h, final_max_h, min_h_mean, max_h_mean

def analysis_person_v4(json_path, image, visual_name, filter=True, tre=0.2, minmin_h=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    joints = []
    H, W, _ = image.shape
    min_h_mean=99999999
    max_h_mean=-1
    min_h, max_h=0, 0
    min_kps_bbox=max_kps_bbox=[]

    all_kps_bbox=[]
    for person_dict in data:
        person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
        confident = np.mean(person_kps[:, 2])
        if filter and confident < tre:
            continue
        patch_name=person_dict['image_id']
        w1,h1,w2,h2=getPosition(patch_name)
        person_kps[:, 0]+=w1
        person_kps[:, 1]+=h1
        all_kps_bbox.append(kps2bbox(person_kps))
    max_h_tre=get_candicate_tre(all_kps_bbox, ratio=0.95)

    all_kps_bbox=[]
    max_h_TopN=10
    max_h_TopN_queue=queue.PriorityQueue()
    pid=0
    for person_dict in data:
        person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
        confident = np.mean(person_kps[:, 2])
        if filter and confident < tre:
            continue
        patch_name=person_dict['image_id']
        w1,h1,w2,h2=getPosition(patch_name)
        person_kps[:, 0]+=w1
        person_kps[:, 1]+=h1
        
        color=np.random.random(3)*255
        kps_bbox=kps2bbox(person_kps)

        width=max([int((kps_bbox[3]-kps_bbox[1])*0.01), 5])
        cv2.rectangle(image, (kps_bbox[0], kps_bbox[1]), (kps_bbox[2], kps_bbox[3]), color=color, thickness=width)

        joints.append(person_kps)
        h_mean=(kps_bbox[1]+kps_bbox[3])/2
        cur_h=max( [int(person_kps[:, 1].max() - person_kps[:, 1].min()), 0])
        if minmin_h > cur_h:
            continue
        
        all_kps_bbox.append(kps_bbox)

        if h_mean < min_h_mean:
            min_h=cur_h
            min_h_mean=max([h_mean,0])
            min_kps_bbox=kps_bbox

        if h_mean > max_h_mean and cur_h > max_h_tre:
            max_h_mean=h_mean
            max_h=cur_h
            max_kps_bbox=kps_bbox
        
        if max_h_TopN_queue.qsize() < max_h_TopN:
            max_h_TopN_queue.put([cur_h, pid])
        else:
            exist_max_h, exist_max_h_pid=max_h_TopN_queue.get()
            if cur_h>exist_max_h:
                max_h_TopN_queue.put([cur_h, pid])
            else:
                max_h_TopN_queue.put([exist_max_h, exist_max_h_pid])
        pid+=1
        
    
    for i in range(len(all_kps_bbox)):
        # for min_kps
        final_min_h=min_h
        if is_h_close(min_kps_bbox, all_kps_bbox[i])>0:
            if (all_kps_bbox[i][3] - all_kps_bbox[i][1]) > final_min_h:
                final_min_h=all_kps_bbox[i][3] - all_kps_bbox[i][1]
        final_max_h=max_h
        if is_h_close(max_kps_bbox, all_kps_bbox[i])>0:
            if (all_kps_bbox[i][3] - all_kps_bbox[i][1]) > final_max_h:
                final_max_h=all_kps_bbox[i][3] - all_kps_bbox[i][1]

    # final_max_h=-1
    # new_h_mean=-1
    # while not max_h_TopN_queue.empty():
    #     cur_h, pid = max_h_TopN_queue.get()
    #     h_mean=all_kps_bbox[pid][3] - all_kps_bbox[pid][1]
    #     if h_mean > new_h_mean:
    #         new_h_mean=h_mean
    #         final_max_h=cur_h
    # max_h_mean=new_h_mean

    cv2.imwrite(visual_name, image)
    return final_min_h, final_max_h, min_h_mean, max_h_mean

def getImg2joints(json_path, filter=False, tre=0.2, use_scale_blocks=True):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img2joints = {}
    for person_dict in data:
        if '.fuse_hidden' in person_dict['image_id']:
            continue
        if person_dict['image_id'] not in img2joints:
            img2joints[person_dict['image_id']] = []
        person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
        if use_scale_blocks:
            info=person_dict['image_id'].split('_')
            origin_size=int(info[-2]) - int(info[-4])
            person_kps[:, :2]=person_kps[:, :2] / 512. * origin_size

        confident = np.mean(person_kps[:, 2])
        if filter and confident < tre:
            continue
        img2joints[person_dict['image_id']].append(person_kps)
    return img2joints

def fetch_joints(json_path, tre=0.2, select_list=[]):
    with open(json_path, 'r') as f:
        data = json.load(f)
    kps_list=[]
    for person_dict in data:

        person_kps = np.array(person_dict['keypoints']).reshape(-1, 3)
        confident = np.mean(person_kps[:, 2])
        if confident < tre:
            continue
        if len(select_list) ==0:
            kps_list.append(person_kps)
        elif person_dict['image_id'] in select_list:
            kps_list.append(person_kps)

    return kps_list

def compare_replace_cps(cur, cur_xy, L, L_xy, tre_ratio=0.3):
    '''
    return: 0 add cur, 1 delete cur, 2 delete pre
    '''
    if len(L) == 0:
        return 0, None, None
    cur = cur[:, :2]
    cur_cps = np.mean(cur, axis=0).reshape(-1, 2)
    L_cps = []
    for i in range(len(L)):
        i_cps = L[i][:, :2]
        L_cps.append(np.mean(i_cps, axis=0))

    data = np.array(L_cps) - cur_cps
    dist = np.sum(data ** 2, 1) ** 0.5
    min_ = np.min(dist)
    cur_x1, cur_y1, cur_x2, cur_y2 = cur_xy
    # cur_size = np.abs(cur_x2 - cur_x1)

    cur_size = np.sqrt(np.sum((cur[5] - cur[12]) ** 2))

    # ref_size = np.abs(cur_x2 - cur_x1) * 0.05
    # if ref_size<cur_size:
    #     cur_size=ref_size

    tre = cur_size * tre_ratio
    if min_ < tre:
        min_id = np.where(dist == min_)[0][0]
        pre_x1, pre_y1, pre_x2, pre_y2 = L_xy[min_id]
        pre_cx, pre_cy = L_cps[min_id][1] - pre_x1, L_cps[min_id][0] - pre_y1
        cur_cx, cur_cy = cur_cps[0][1] - cur_x1, cur_cps[0][0] - cur_y1
        pre_size = np.abs(pre_x2 - pre_x1)
        if pre_x1 < cur_x1:  # up
            last_d = pre_size - pre_cx
            cur_d = cur_cx
            if cur_d > last_d:
                L = L[:min_id] + L[min_id + 1:]
                L_xy = L_xy[:min_id] + L_xy[min_id + 1:]
                return 2, L, L_xy
            else:
                return 1, None, None
        elif pre_x1 == cur_x1:  # left
            last_d = pre_size - pre_cy
            cur_d = cur_cy
            if cur_d > last_d:
                L = L[:min_id] + L[min_id + 1:]
                L_xy = L_xy[:min_id] + L_xy[min_id + 1:]
                return 2, L, L_xy
            else:
                return 1, None, None
        else:
            print("pre_x1 > cur_x1")
    else:
        return 0, None, None


def merge(img2joints, position_path, row_col_path, name_pre, method='joints bbox'):
    position = np.load(position_path)
    row_col = np.load(row_col_path)
    if method == 'joints bbox':
        compare_replace = compare_replace_by_bbox
    elif method == 'joints cps':
        compare_replace = compare_replace_cps

    x2row_dict = {}
    y2col_dict = {}

    kps_list = []
    kps_xy_list = []
    last_row = -1
    cur_row_kps = []
    last_row_kps = []
    cur_row_xy = []
    last_row_xy = []
    cur_kps = []
    cur_xy = []
    last_size = 0
    total_kps_num = 0
    total_kps = []

    for i in range(position.shape[0]):
        x1, y1, x2, y2 = position[i].astype(int)

        row, col = row_col[i].astype(int)
        x2row_dict[str(x1) + '_' + str(x2)] = row
        y2col_dict[str(y1) + '_' + str(y2)] = col
        cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(x1) + '_' + str(y1) + '_' + str(
            x2) + '_' + str(
            y2) + '.jpg'
        if last_row < row:  # new line
            kps_list.append(cur_row_kps)
            kps_xy_list.append(cur_row_xy)
            last_row += 1
            last_row_kps = cur_row_kps
            cur_row_kps = []
            last_row_xy = cur_row_xy
            cur_row_xy = []
        if i == position.shape[0] - 1:
            kps_list.append(cur_row_kps)
            kps_xy_list.append(cur_row_xy)
        cur_kps = []
        cur_xy = []

        if cur_name in img2joints:
            kps = img2joints[cur_name]

            for k_id, person_kps in enumerate(kps):
                total_kps_num += 1
                person_kps[:, 0] += y1
                person_kps[:, 1] += x1
                cur = person_kps
                total_kps.append(cur)

                add_flag = 0
                # left
                r1, a, b = compare_replace(cur, (x1, y1, x2, y2), cur_row_kps, cur_row_xy)
                if r1 == 0:  # add cur
                    add_flag = 1

                elif r1 == 1:  # delete cur
                    pass
                elif r1 == 2:  # delete pre
                    add_flag = 2
                    cur_row_kps, cur_row_xy = a, b
                    cur_kps.append(cur)
                    cur_xy.append((x1, y1, x2, y2))

                # up
                r2, c, d = compare_replace(cur, (x1, y1, x2, y2), last_row_kps, last_row_xy)
                if r2 == 0:  # not relate
                    pass
                elif r2 == 1:  # delete cur
                    add_flag = 0
                elif r2 == 2:  # delete pre
                    last_row_kps, last_row_xy = c, d
                    kps_list[-1] = last_row_kps
                    kps_xy_list[-1] = last_row_xy
                    # if r1 == 1:
                    #     add_flag=3
                    #     cur_cps.append(cur)
                    #     cur_xy.append((x1, y1, x2, y2))

                if add_flag == 1:
                    cur_kps.append(cur)
                    cur_xy.append((x1, y1, x2, y2))

            cur_row_kps = cur_row_kps + cur_kps
            cur_row_xy = cur_row_xy + cur_xy

    final_kps = []
    final_kps_xy = []
    for i in range(len(kps_list)):
        final_kps = final_kps + kps_list[i]
        final_kps_xy = final_kps_xy + kps_xy_list[i]

    return final_kps, final_kps_xy, total_kps, x2row_dict, y2col_dict


def cps_filter(final_kps, final_kps_xy, tre_ratio=0.1):
    new_final_kps = []
    new_final_kps_xy = []
    num = len(final_kps)
    using_flag = np.zeros(num)
    final_center_points = []
    final_size = []
    final_confient = []
    for person_kps in final_kps:
        final_confient.append(np.squeeze(np.mean(person_kps[:, 2])))
        i_kps = person_kps[:, :2]
        final_center_points.append(np.mean(i_kps[[5, 6, 11, 12]], axis=0))
        final_size.append(np.sqrt(np.sum((i_kps[1] - i_kps[2]) ** 2)))  # distance between eyes

    X = np.array(final_center_points).reshape(-1, 2)
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (X.shape[0], 1))
    D = H + H.T - 2 * G  # The Distance Matrix
    D = D ** 0.5
    for i in range(num):
        if using_flag[i] == 1:
            continue
        using_flag[i] = 1
        cur_x1, cur_y1, cur_x2, cur_y2 = final_kps_xy[i]
        cur_size = final_size[i]
        tre = tre_ratio * cur_size
        cur_D = D[i]
        cur_D[i] = 999999

        min_index = 0
        min_value = 999999
        for ii, value in enumerate(cur_D):
            if value < min_value:
                min_value = value
                min_index = ii

        if min_value < tre and using_flag[min_index] == 0:
            if final_confient[i] > final_confient[min_index]:
                new_final_kps.append(final_kps[i])
                new_final_kps_xy.append(final_kps_xy[i])
            else:
                new_final_kps.append(final_kps[min_index])
                new_final_kps_xy.append(final_kps_xy[min_index])
            using_flag[min_index] = 1
        else:
            new_final_kps.append(final_kps[i])
            new_final_kps_xy.append(final_kps_xy[i])

    return new_final_kps, new_final_kps_xy


# def merge_joints(joint_json, part_path, name_pre, visual_flag=False, scene_image_path='', check_name='check.jpg', confident_tre=0.4):
#     img2joints = getImg2joints(joint_json, filter=True, tre=confident_tre)
#     position_path = os.path.join(part_path, 'position.npy')
#     row_col_path = os.path.join(part_path, 'row_col.npy')
#     final_kps, final_kps_xy, total_kps, x2row_dict, y2col_dict = merge(img2joints, position_path, row_col_path,
#                                                                        name_pre, method='joints cps')
#     print('after merge, kps_num %d / %d' % (len(final_kps), len(total_kps)))
#     # final_kps, final_kps_xy=cps_filter(final_kps, final_kps_xy, tre_ratio=1)
#     # print('after filter, kps_num %d / %d' %(len(final_kps), len(total_kps)))

#     if visual_flag:
#         scene_image = cv2.imread(scene_image_path)
#         for per_kps in final_kps:
#             color = np.array(np.random.rand(3)) * 255
#             per_kps = per_kps[:, :2]
#             circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.1)

#             for coord in per_kps:
#                 x_coord, y_coord = int(coord[0]), int(coord[1])

#                 cv2.circle(scene_image, (x_coord, y_coord), circle_size, color, -1)
#         cv2.imwrite(check_name, scene_image)

#     return final_kps

def merge_joints(json_folder, part_path, name_pre, visual_flag=False, scene_image_path='', check_name='check.jpg', confident_tre=0.4, use_scale_blocks=True):
    
    if use_scale_blocks:
        joint_json=os.path.join(json_folder, 'alphapose-results-scale.json')
    else:
        joint_json=os.path.join(json_folder, 'alphapose-results-origin.json')
    img2joints = getImg2joints(joint_json, filter=True, tre=confident_tre, use_scale_blocks=use_scale_blocks)
    position_path = os.path.join(part_path, 'position.npy')
    row_col_path = os.path.join(part_path, 'row_col.npy')
    final_kps, final_kps_xy, total_kps, x2row_dict, y2col_dict = merge(img2joints, position_path, row_col_path,
                                                                       name_pre, method='joints cps')
    # print('after merge, kps_num %d / %d' % (len(final_kps), len(total_kps)))

    if visual_flag:
        scene_image = cv2.imread(scene_image_path)
        for per_kps in final_kps:
            color = np.array(np.random.rand(3)) * 255
            per_kps = per_kps[:, :2]
            circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.1)

            for coord in per_kps:
                x_coord, y_coord = int(coord[0]), int(coord[1])

                cv2.circle(scene_image, (x_coord, y_coord), circle_size, color, -1)
        cv2.imwrite(check_name, scene_image)

    return final_kps



def joints2bbox(person_kps):  # kps bbox
    w1 = min(person_kps[:, 0])
    w2 = max(person_kps[:, 0])
    h1 = min(person_kps[:, 1])
    h2 = max(person_kps[:, 1])
    return (w1, h1, w2, h2)


def compute_iou(pos1, pos2):  
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
        return inter /(area_sum - inter)

def write_json(name, data):
    with open(name, 'w') as wf:
        json.dump(data, wf)

def have_person_by_joints(all_joints, part_path, name_pre, save_name=''):
    position=np.load(os.path.join(part_path, 'position.npy'))
    row_col=np.load(os.path.join(part_path, 'row_col.npy'))
    select_patch_names=[]

    for i in range(position.shape[0]):
        h1, w1, h2, w2 = position[i].astype(int)
        row, col = row_col[i].astype(int)
        cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(h1) + '_' + str(w1) + '_' + str(
            h2) + '_' + str(
            w2) + '.jpg'
        patch_as_bbox=(w1, h1, w2, h2)
        for kps in all_joints: 
            kps_bbox = joints2bbox(kps)
            iou_for_kps = compute_iou(patch_as_bbox, kps_bbox)
            if iou_for_kps > 0:  # inner or overlap
                select_patch_names.append(cur_name)
                break
    if save_name != '':
        write_json(save_name, select_patch_names)
    print('have_person_image %d/ %d' %(len(select_patch_names), position.shape[0]))

    return

