import cv2, os
from lib.utils import read_json, write_json

def kps2bbox(kps):
    w1, w2 = kps[:, 0].min(), kps[:, 0].max()
    h1, h2 = kps[:, 1].min(), kps[:, 1].max()
    return [w1, h1, w2, h2]

def compute_iou(bbox1, bbox2):  # iou for kps
    # print(pos1, pos2)
    left1, top1, right1, down1 = bbox1
    left2, top2, right2, down2 = bbox2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    area_sum = area1 + area2

    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / (area_sum - inter)
    
def is_people(patch_folder, joints, save_name):
    patch_name_list=os.listdir(patch_folder)
    have_people_name_list=[]
    for patch_name in patch_name_list:
        infos=patch_name.split('_')
        h1, h2 = int(infos[-4]), int(infos[-2])
        w1, w2 = int(infos[-3]), int(infos[-1][:-4])
        patch_bbox=[w1,h1,w2,h2]
        have_people_flag=False
        for id, p_kps in enumerate(joints):
            if p_kps is not None:
                p_kps=p_kps.reshape(17, 3)
                p_bbox=kps2bbox(p_kps)
                iou_for_kps = compute_iou(patch_bbox, p_bbox)
                if iou_for_kps > 0:  # inner or overlap
                    have_people_flag=True
                    break
        if have_people_flag:
            have_people_name_list.append(patch_name)
    # print('is_people', len(patch_name_list), len(have_people_name_list))
    write_json(save_name, have_people_name_list)
    
def scale_patches(source_image_root, target_image_root):
    patch_image_list=os.listdir(source_image_root)
    os.makedirs(target_image_root, exist_ok=True)
    for patch_image_name in patch_image_list:
        image=cv2.imread(os.path.join(source_image_root, patch_image_name))
        image_scale=cv2.resize(image, (512, 512))
        cv2.imwrite(os.path.join(target_image_root, patch_image_name), image_scale)