import torch
import numpy as np

import sys, os

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import constants
from config import args

def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float().to(kp2ds.device)
    img_pad_size, crop_trbl, pad_trbl, data_scale = offsets[:, :
                                                2], offsets[:,
                                                            2:6], offsets[:,
                                                                          6:10], offsets[:, 10]                                                                    
    leftTop = torch.stack(
        [crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]],
        1)
    data_scale=data_scale.unsqueeze(-1)
    leftTop=leftTop * data_scale
    orign_size=img_pad_size * data_scale

    kp2ds_on_orgimg = (
        kp2ds + 1) * orign_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    return kp2ds_on_orgimg


def vertices_kp3d_projection(outputs,
                             meta_data=None,
                             presp=args().model_version > 3):
    params_dict, vertices, j3ds = outputs['params'], outputs['verts'], outputs[
        'j3d']
    verts_camed = batch_orth_proj(vertices,
                                  params_dict['cam'],
                                  mode='3d',
                                  keep_dim=True)
    pj3d = batch_orth_proj(j3ds, params_dict['cam'], mode='2d')
    projected_outputs = {'verts_camed': verts_camed, 'pj2d': pj3d[:, :, :2]}

    if meta_data is not None:
        projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(
            projected_outputs['pj2d'], meta_data['offsets'])
    return projected_outputs


def compute_angle(v1,v2):
    '''
    v1: m,3
    v2: 1,3 or m,3
    '''
    v=torch.sum(v1*v2, dim=1)
    v_norm=torch.norm(v1, dim=1)*torch.norm(v2, dim=1)
    cos_v=v/v_norm
    angle=torch.acos(cos_v)
    return angle

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def rotate_ground_Matrix(ground):
    N=ground[:, :3]
    fenmu=torch.norm(N, dim=1)
    N=N/fenmu.unsqueeze(-1)

    Nx=torch.zeros_like(N)
    Nx[:, [1,2]]=N[:, [1,2]]
    Nz=torch.zeros_like(N)
    Nz[:, [0,1]]=N[:, [0,1]]
    v_y=torch.tensor([[0, -1., 0]], device=ground.device)

    angle_x = compute_angle(Nx, v_y)
    c1=ground[:, 2].gt(0)
    angle_x=angle_x * (c1*2-1)

    angle_x=angle_x.unsqueeze(-1)
    padding=torch.zeros_like(angle_x)
    angle_x_v=torch.cat([angle_x, padding, padding], dim=-1)
    rotate_M1=batch_rodrigues(angle_x_v)


    angle_z = compute_angle(Nz, v_y)
    c2=ground[:, 0].lt(0)
    angle_z=angle_z* (c2*2-1)

    angle_z=angle_z.unsqueeze(-1)
    padding=torch.zeros_like(angle_z)

    angle_z_v=torch.cat([padding, padding, angle_z], dim=-1)
    rotate_M2=batch_rodrigues(angle_z_v)
    rotate_M=torch.matmul(rotate_M2, rotate_M1)

    # check=rotate_M @ N.unsqueeze(-1)

   
    return rotate_M



def vertices_kp3d_projection_ground(outputs, meta_data):
    '''
    achieve projection on ground.
    :param outputs:
    :param meta_data:
    :return:
    '''

    params_dict, vertices, j3ds = outputs['params'], outputs['verts'], outputs[
        'j3d']

    j3ds_17=outputs['j3d_17']
    j3ds=torch.cat([j3ds, j3ds_17], dim=1)


    predict_len = params_dict['hvip2d'] * 1.5 # m, 1   (0,1)

    predict_tc_offset = (params_dict['tc_offset'] * 2 - 1) * 0.25 # m,2 

    predict_delta3d=(params_dict['delta3d']*2 -1) * 0.4 # m, 1


    # m,4          m,3,3       m,2            m,2       m,5
    ground, batch_camK, patch_leftTop, batch_dist = meta_data['ground'], \
                                                    meta_data['camK'], \
                                                    meta_data['patch_leftTop'], \
                                                    meta_data['dist']

    offsets = meta_data['offsets']
    offsets = offsets.float().to(vertices.device)
    img_pad_size, crop_trbl, pad_trbl, data_scale = offsets[:, :
                                                2], offsets[:,
                                                            2:6], offsets[:,
                                                                          6:10], offsets[:, 10]
    data_scale=data_scale.unsqueeze(-1)
    leftTop = torch.stack(
        [crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]],
        1)
    orign_size=img_pad_size * data_scale
    leftTop=leftTop * data_scale

    N=ground[:, :3]
    N=N / torch.norm(N, p=2, dim=1).unsqueeze(-1)
    X_n, Y_n, Z_n=N[:, 0], N[:, 1], N[:, 2]
    A, B, C, D = ground[:, 0], ground[:, 1], ground[:, 2], ground[:, 3]
    fx =  batch_camK[:, 0, 0]
    fy = batch_camK[:, 1, 1]
    cx = batch_camK[:, 0, 2]
    cy = batch_camK[:, 1, 2]


    torso_center=outputs['centers_pred']/ args().centermap_size + predict_tc_offset # 0-1 范围的
    tcs=torso_center * orign_size + leftTop + patch_leftTop
    tc=torso_center * 2 -1 # (-1, 1)
    u, v = tcs[:, 0], tcs[:, 1]

    N=N.double()
    temp=torch.matmul(batch_camK, N.unsqueeze(-1))[:, :, 0]
    vp = temp[:, :2] / temp[:, 2].unsqueeze(-1) # m,2

    vp_direction=vp-tcs
    vp_direction=vp_direction / torch.norm(vp_direction, dim=1).unsqueeze(-1)
    hvip_position_2d= predict_len * vp_direction + tc

    hvip_position_2d_orign = (hvip_position_2d +
                              1) * orign_size / 2 + leftTop

    hvip_position_2d_orign_in_scene = patch_leftTop + hvip_position_2d_orign  # m, 2

    hvip_position_2d_orign_in_scene = torch.cat([
        hvip_position_2d_orign_in_scene,
        torch.ones_like(hvip_position_2d_orign_in_scene[:, 0])[:, None]
    ],
                                                axis=-1)  # m,3
    fx2_reci = 1. / batch_camK[:, 0, 0]
    fy2_reci = 1. / batch_camK[:, 1, 1]

    new_depth = -ground[:, 3] / (
            ground[:, 0] * fx2_reci * (hvip_position_2d_orign_in_scene[:, 0] - batch_camK[:, 0, 2]) \
            + ground[:, 1] * fy2_reci * (hvip_position_2d_orign_in_scene[:, 1] - batch_camK[:, 1, 2]) \
            + ground[:, 2])
    new_depth = new_depth.unsqueeze(-1)  # m,1
    K_inv = torch.inverse(batch_camK)

    temp = (hvip_position_2d_orign_in_scene * new_depth).unsqueeze(-1)
    hvip_position_3d = torch.matmul(K_inv, temp).squeeze(-1)  # m,3


    X0, Y0, Z0 = hvip_position_3d[:, 0], hvip_position_3d[:, 1], hvip_position_3d[:, 2]
    m1=(u-cx) / fx
    m2=(v-cy) / fy
    H=(m2 * Z0 - Y0)/ (Y_n - m2 * Z_n)


    H=H.unsqueeze(-1)

    tcs_3d = hvip_position_3d.float() +  H * N  + predict_delta3d
    translation = tcs_3d
    
    # # # select the root_trans
    method = 'meanOfTorso'
    if method not in [
            'meanOfAnkles',
            'no', 'meanOfTorso'
    ]:
        print('method=', method, 'it is illegal')
    if method =='meanOfTorso':
        root_trans = (j3ds[:, 16].unsqueeze(1) + j3ds[:, 17].unsqueeze(1) + j3ds[:, 45].unsqueeze(1) + j3ds[:, 46].unsqueeze(1)) / 4
    if method == 'meanOfAnkles':
        # # way 1: using the mean of jont ankles
        root_trans = (j3ds[:, 7].unsqueeze(1) + j3ds[:, 8].unsqueeze(1)) / 2
    
    if method == 'no':
        root_trans = torch.zeros_like(j3ds[:, 0, :]).unsqueeze(1)

    correct_smpl_scale=torch.tensor(args().correct_smpl_scale).to(vertices.device)
    vertices_rootCenter = (vertices - root_trans) * correct_smpl_scale
    j3d_rootCenter = (j3ds - root_trans) * correct_smpl_scale

    verts_camed = vertices_rootCenter + translation.unsqueeze(1)
    rotation_Is = torch.eye(3).unsqueeze(0).repeat(
        vertices_rootCenter.shape[0], 1, 1).to(vertices_rootCenter.device)

    pj2d = perspective_projection_cam(j3d_rootCenter, rotation_Is, translation,
                                      batch_camK, batch_dist)
    zero_point = torch.zeros_like(root_trans)
    new_hvip2d = perspective_projection_cam(zero_point, rotation_Is,
                                            hvip_position_3d.float(),
                                            batch_camK, batch_dist).reshape(
                                                pj2d.shape[0], 2)

    pj2d = pj2d - patch_leftTop.unsqueeze(1) - leftTop.unsqueeze(1)
    new_hvip2d = new_hvip2d - patch_leftTop - leftTop

    new_hvip2d = new_hvip2d / orign_size * 2 - 1
    old_hvip2d = hvip_position_2d
    pj2d = pj2d / orign_size.unsqueeze(1) * 2 - 1

    # compute dist to ground
    dist_ground = dist_point_ground(ground, verts_camed)

    # for root_cam
    j3d_camed = j3d_rootCenter + translation.unsqueeze(1) #m, 54, 3
    hip_index=[45, 46]
    hips=j3d_camed[:, hip_index,:]
    root_cam=hips.mean(1)
    final_trans = -root_trans.squeeze(1) + translation

    projected_outputs = {
        'verts_camed': verts_camed,
        'pj2d': pj2d,
        'pj2d_backup': pj2d.clone(),
        'old_hvip2d': old_hvip2d.clone(),
        'new_hvip2d': new_hvip2d.clone(),
        'hvip3d': hvip_position_3d,
        'dist_ground': dist_ground,
        'final_trans': final_trans,
        'pred_root_cam': root_cam,
        'torso_center': torso_center * 2 -1.,
        'tc3d': tcs_3d
    }  # m,6890,2   m, 54, 2
    
    projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(
        projected_outputs['pj2d'], meta_data['offsets'])
    hvip2d_org=convert_kp2d_from_input_to_orgimg(old_hvip2d.unsqueeze(1), meta_data['offsets'])
    projected_outputs['hvip2d_org']=hvip2d_org.squeeze(1)
    projected_outputs['hvip2d_in_scene']=hvip_position_2d_orign_in_scene[:, :2]

    return projected_outputs


def dist_point_ground(ground, points):
    '''
    ground: m,4
    points: m, N, 3
    '''
    # fenzi m,N         fenmu  m, 1
    fenzi = ground[:,
                   0].unsqueeze(-1) * points[:, :, 0] + ground[:, 1].unsqueeze(
                       -1) * points[:, :, 1] + ground[:, 2].unsqueeze(
                           -1) * points[:, :, 2] + ground[:, 3].unsqueeze(-1)
    fenmu = torch.sqrt(ground[:, 0]**2 + ground[:, 1]**2 +
                       ground[:, 2]**2).unsqueeze(-1)

    dist = fenzi / fenmu
    return dist  #m, N


def batch_orth_proj(X, camera, mode='2d', keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:, :, :2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:, :, 2].unsqueeze(-1)], -1)
    return X_camed


def project_2D(kp3d, cams, keep_dim=False):
    d, f, t = cams[0], cams[1], cams[2:].unsqueeze(0)
    pose2d = kp3d[:, :2] / (kp3d[:, 2][:, None] + d)
    pose2d = pose2d * f + t
    if keep_dim:
        kp3d[:, :2] = pose2d
        return kp3d
    else:
        return pose2d


# rotation: rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_joints.device)
# translation: translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(pred_joints.device)
def perspective_projection(points, rotation, translation, focal_length,
                           camera_center, batch_dist):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    K = K.float()
    projected_points = projected_points.float()
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    # X = torch.from_numpy(X)
    # camera_params = torch.from_numpy(camera_params)
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    # XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    XX = X[..., :2] / X[..., 2:]
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat(
        (r2, r2**2, r2**3), dim=len(r2.shape) - 1),
                           dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def perspective_projection_cam(points, rotation, translation, batch_camK,
                               batch_dist):
    # def perspective_projection_cam(points, rotation, translation,batch_camK):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        batch_camk (bs, 3,3): Camera center
    """

    K = batch_camK  # (m,3,3)
    # print('K', K.cpu().detach().numpy())

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # print('net j3d mean', torch.mean(points, dim=1).cpu().detach().numpy())
    # in_cam 应为 （m，9）
    batch_size = K.shape[0]
    index_in_cam = torch.from_numpy(np.arange(batch_size)).to(points.device)
    in_cam = torch.zeros((batch_size, 9)).to(points.device)
    for i, camK, dist in zip(index_in_cam, batch_camK, batch_dist):
        in_cam[i, 0] = camK[0, 0]
        in_cam[i, 1] = camK[1, 1]
        in_cam[i, 2] = camK[0, 2]
        in_cam[i, 3] = camK[1, 2]
        in_cam[i, 4:] = dist

    projected_points = project_to_2d(points, in_cam)

    return projected_points[:, :, :2]
