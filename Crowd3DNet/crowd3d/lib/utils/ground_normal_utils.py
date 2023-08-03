import numpy as np
import cv2
import time


def foot2d_to_foot3d(foot2d_in_scene, camM, ground):
    '''
    foot2d_in_scene   (m,2)
    camM   (3,3)
    ground   (4,)
    '''
    foot2d_in_scene = np.concatenate([foot2d_in_scene, np.ones((foot2d_in_scene.shape[0], 1))], axis=-1) # m, 3
    fx_inverse, fy_inverse = 1/camM[0,0], 1/camM[1, 1]
    new_depth = -ground[3] / (
            ground[0] * fx_inverse * (foot2d_in_scene[:, 0] - camM[0, 2]) \
            + ground[1] * fy_inverse * (foot2d_in_scene[:, 1] - camM[1, 2]) \
            + ground[2])
    K_inv = np.linalg.inv(camM)
    temp = foot2d_in_scene * np.expand_dims(new_depth, -1)
    foot3d = (K_inv @ temp.T).T # m,3
    return foot3d

def convert(foot2d_in_scene, camM, ground):
    foot3ds=foot2d_to_foot3d(foot2d_in_scene, camM, ground) # m, 3
    N=ground[:3]
    if N[1] > 0:
        N=-N
    mo=np.linalg.norm(N)
    N_norm=N/mo # 3
    N_norm=np.expand_dims(N_norm, 0)
    foot3ds_trans=foot3ds + N_norm
    temp=foot3ds_trans @ camM.T
    foot2ds_transed=temp[:, :2] / np.expand_dims(temp[:,2], -1)
    return foot2ds_transed

def get_xx_yy(h, w, delta_h=0, delta_w=0):
    res=[]
    for i in range(h):
        y_col=np.ones((w, 1), dtype=int) * i + delta_h
        x_col=np.arange(w).reshape(w, 1) + delta_w
        x_y=np.concatenate([x_col, y_col], -1)
        res.append(x_y)
    res=np.concatenate(res, axis=0)
    return res

def get_xx_yy2(h, w, delta_h=0, delta_w=0):
    w_index=np.array(range(w)).reshape(1, w)
    h_index=np.array(range(h)).reshape(h, 1)

    w_ones=np.ones((1, w))
    h_ones=np.ones((h, 1))

    w_grid= (h_ones @ w_index).reshape(-1, 1) + delta_w
    h_grid=(h_index @ w_ones).reshape(-1, 1)+ delta_h
    res=np.concatenate([w_grid, h_grid], axis=-1).astype(int)
    return res

def visual_GN_patch(image, image_trans, interval=1, save_path=None):
    h, w, _ = image.shape
    color=np.random.random(3)*255
    for h_i in range(0, h, interval):
        for w_i in range(0, w, interval):
            pt1=(w_i,h_i)
            pt2_w, pt2_h = int(w_i+image_trans[h_i, w_i,0]), int(h_i+image_trans[h_i, w_i, 1])
            # if pt2_w < 0 or pt2_w >=w or pt2_h <0 or pt2_h>=h:
            #     continue
            pt2=(pt2_w, pt2_h)
            cv2.arrowedLine(image,pt1, pt2, color=color, thickness=2, tipLength=0.2)
    if save_path is not None:
        cv2.imwrite(save_path, image)

def get_imageGn_by_project(camMats, ground, height, width, delta_h=0, delta_w=0, norm=True):
    xx_yy_grid = get_xx_yy(height, width, delta_h=delta_h, delta_w=delta_w)
    xx_yy_grid_GN_trans=convert(xx_yy_grid, camMats, ground) - xx_yy_grid
    if norm:
        xx_yy_grid_GN_trans_norm=np.expand_dims(np.linalg.norm(xx_yy_grid_GN_trans, axis=-1), -1)
        xx_yy_grid_GN_trans=xx_yy_grid_GN_trans / xx_yy_grid_GN_trans_norm
    image_trans=xx_yy_grid_GN_trans.reshape(height, width, 2)
    return image_trans

def get_imageGn_by_vp(camMats, ground, height, width, delta_h=0, delta_w=0, norm=True):
    xx_yy_grid = get_xx_yy(height, width, delta_h=delta_h, delta_w=delta_w)
    xx_yy_grid_GN_trans = convert_by_vp(xx_yy_grid, camMats, ground) 
    if norm:
        xx_yy_grid_GN_trans_norm = np.expand_dims(
            np.linalg.norm(xx_yy_grid_GN_trans, axis=-1), -1)
        xx_yy_grid_GN_trans = xx_yy_grid_GN_trans / xx_yy_grid_GN_trans_norm
    image_trans = xx_yy_grid_GN_trans.reshape(height, width, 2)
    return image_trans


def convert_by_vp(foot2d_in_scene, camM, ground):
    N = ground[:3]
    if N[1] > 0:
        N = -N
    mo = np.linalg.norm(N)
    N_norm = N/mo  # 3
    N_norm = np.expand_dims(N_norm, 0)

    temp=N_norm @ camM.T
    vp = temp[:, :2] / np.expand_dims(temp[:, 2], -1)
    trans=vp - foot2d_in_scene

    return trans