import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from config import args
import numpy as np
import torch
import cv2
import constants

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70, # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    },
]

h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S2': [
        {},
        {},
        {},
        {},
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}




def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    # print('qvec',qvec.dtype)
    # print('v',v.dtype)
    qvec = qvec.to(torch.float32)
    v = v.to(torch.float32)
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    #print('q',q)
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)



def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    X = torch.from_numpy(X)
    camera_params = torch.from_numpy(camera_params)
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

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2

    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    X = torch.from_numpy(X)
    camera_params = torch.from_numpy(camera_params)
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f*XX + c



def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    #axyz = augment(xyzs[:3])
    axyz = augment(xyzs)
    return np.linalg.svd(axyz)[-1][-1, :]

def get_ground_from_world(R,t):
    '''
    input:
        R (3,3)
        T (3,)
        cam_i: for save
    output:
        ground: (4,)
    '''
    axyz = np.array([[0,0,1000],[1000,0,1000],[1000,0,0],[500,0,1500]])
    axyz_cam = convert_wc_to_cc(axyz,R,t)
    m = estimate(axyz_cam)

    # fig = plt.figure()
    # ax = mplot3d.Axes3D(fig)
    #
    # def plot_plane(a, b, c, d):
    #     xlim = ax.get_xlim()
    #     ylim = ax.get_ylim()
    #     xx,yy = np.meshgrid(np.arange(xlim[0], xlim[1]),
    #               np.arange(ylim[0], ylim[1]))
    #     #xx, yy = np.mgrid[:10, :10]
    #     return xx, yy, (-d - a * xx - b * yy) / c
    #
    # ax.scatter3D(axyz.T[0], axyz.T[1], axyz.T[2])
    # ax.scatter3D(axyz_cam.T[0], axyz_cam.T[1], axyz_cam.T[2],c = 'g')
    # a, b, c, d = m
    # xx, yy, zz = plot_plane(a, b, c, d)
    # ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))
    # save_path = os.path.join("/media/panda_data/lock_you_on_the_ground/restult/ground_ransac/panoptic/")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # print('����·����',save_path)
    # plt.savefig(os.path.join(save_path,"real_ground"+ str(cam_i) + ".png"))
    # #plt.show()
    # plt.close()

    return m

def get_ground_from_world_pan(R,t):
    '''
    input:
        R (3,3)
        T (3,1)
    output:
        ground: (4,)
    '''
    axyz = np.array([[0,0,100],[100,0,100],[100,0,0],[50,0,150]])
    axyz_cam = (R @ axyz.T + t).T  / 100  #cm to m
    m = estimate(axyz_cam)
    return m 

def convert_cc_to_wc(joints_cam,R,T):
    '''
    input:
        joint_cam (N,num_joints,3)
        R (3,3)
        T (3,)
    output:
        joint_world
    '''
    joints_cam = np.asarray(joints_cam)
    # 相机坐标系 -> 世界坐标系
    joints_world = []
    for joint_cam in joints_cam:
        joint_world = np.dot(np.linalg.inv(R), (joint_cam-T).T).T
        joints_world.append(joint_world)
    return np.array(joints_world)

def convert_wc_to_cc(joints_world,R,T):
    '''
    input:
        joint_world (N,num_joints,3)
        R (3,3)
        T (3,)
    output:
        joint_cam  (N,num_joints,3)
    '''
    joints_world = np.asarray(joints_world)
    # 世界坐标系 -> 相机坐标系
    joints_cam = []
    for joint_world in joints_world:
        joint_cam = np.dot(R, joint_world.T).T + T
        joints_cam.append(joint_cam)
    return np.array(joints_cam)

def map_kps(joint_org, maps=None):
    kps = joint_org[maps].copy()
    kps[maps == -1] = -2.
    return kps

def get_lock_infos(info,R_t,K,camDists,kp3ds):
    #align with camkp3d
    camkp3d = info['kp3d_mono'].reshape(-1,3).copy()  #(32,3)
    mapper_32_to_torso = constants.joint_mapping(constants.H36M_32,constants.torso)
    camkp3d_32_to_torso = map_kps(camkp3d, maps=mapper_32_to_torso)
    mapper_54_to_torso = constants.joint_mapping(constants.SMPL_ALL_54,constants.torso)
    kp3ds_54_to_32 = map_kps(kp3ds.squeeze(), maps=mapper_54_to_torso)
    trans_T = (camkp3d_32_to_torso - kp3ds_54_to_32).mean(axis=0)
    # trans_T = camkp3d[3,:] - kp3ds[0,8,:]
    
    kp3ds_new_cam = kp3ds[0,:] + trans_T
    # print('verts_new_cam:',verts_new_cam.shape)  #(6890,3)
    #convert to world coordinate
    # R_t = h36m_cameras_extrinsic_params[subject_id][cam_view_id]
    R = np.array(R_t['orientation'])
    T = np.array(R_t['translation']) /1000
    
    kp3ds_org = kp3ds_new_cam.copy()
    
    kp3ds_float = np.squeeze(kp3ds_org)
    
    kp3ds_world = camera_to_world(kp3ds_float,R, T)
        
    #unified coordinate
    rotate_v=np.array([-np.pi/2, 0, 0])
    rotate_m=cv2.Rodrigues(rotate_v)[0]
    # print('rotate_m',rotate_m)
    
    kp3ds_world_new = (rotate_m @ kp3ds_world.T).T
    # print('verts_world_new:',verts_world_new.shape)  #(6890,3)
        
    
    #project choosed vert to picture
    # in_cam = np.array([K[0,0],K[1,1],K[0,2],K[1,2],camDists[0],camDists[1],camDists[4],camDists[2],camDists[3]])
    in_cam = np.array([K[0,0],K[1,1],K[0,2],K[1,2],camDists[0],camDists[1],camDists[2],camDists[3],camDists[4]])
    in_cam = np.expand_dims(in_cam,axis=0)
   
    #way unified SMPL 0
    vert_choose = kp3ds_world_new[0]

    vert_min = vert_choose.copy()
    vert_min[1] = 0
    vert_world = (np.linalg.inv(rotate_m) @ vert_min.T).T
    vert_cam = world_to_camera(np.expand_dims(vert_world,axis=0),R, T)
    lock_2d = project_to_2d(np.expand_dims(vert_cam,axis=0),in_cam).numpy()
    lock_2d = np.squeeze(lock_2d,axis=0)
    # print('lock_2d:',lock_2d.shape) #(1,2)

    trans_end_world = (np.linalg.inv(rotate_m) @ vert_choose.T).T
    trans_end_cam = world_to_camera(np.expand_dims(trans_end_world,axis=0),R, T)
    trans_cam = trans_end_cam - vert_cam  #trans_start = vert_choose trans_end = vert_min
    trans_cam_d = np.linalg.norm(trans_cam)
    trans_need = trans_cam - kp3ds[0,0]
    trans_need_d = np.linalg.norm(trans_need)
    
    
    #ground
    point_unified = np.array([[0,0,0],[100,0,0],[0,0,100]])
    point_world = (np.linalg.inv(rotate_m) @ point_unified.T).T
    point_cam = world_to_camera(point_world,R, T)
    ground_cam = estimate(point_cam)

    lock_info = {'lock_2d':lock_2d,'lock_3d':vert_cam,\
        'ground':ground_cam,'trans_cam':trans_cam,'trans_cam_d':trans_cam_d,\
        'trans_need':trans_need,'trans_need_d':trans_need_d}
    
    return lock_info


def get_lock_infos_torso(info, R_t, K, camDists, kp3ds):
    # align with camkp3d
    camkp3d = info['kp3d_mono'].reshape(-1, 3).copy()  # (32,3)
    mapper_32_to_torso = constants.joint_mapping(constants.H36M_32, constants.torso)
    camkp3d_32_to_torso = map_kps(camkp3d, maps=mapper_32_to_torso)
    mapper_54_to_torso = constants.joint_mapping(constants.SMPL_ALL_54, constants.torso)
    kp3ds_54_to_32 = map_kps(kp3ds.squeeze(), maps=mapper_54_to_torso)
    trans_T = (camkp3d_32_to_torso - kp3ds_54_to_32).mean(axis=0)
    # trans_T = camkp3d[3,:] - kp3ds[0,8,:]

    kp3ds_new_cam = kp3ds[0, :] + trans_T
    # print('verts_new_cam:',verts_new_cam.shape)  #(6890,3)
    # convert to world coordinate
    # R_t = h36m_cameras_extrinsic_params[subject_id][cam_view_id]
    R = np.array(R_t['orientation'])
    T = np.array(R_t['translation']) / 1000

    kp3ds_org = kp3ds_new_cam.copy()

    kp3ds_float = np.squeeze(kp3ds_org)

    kp3ds_world = camera_to_world(kp3ds_float, R, T)
    camkp3d_world = camera_to_world(camkp3d, R, T)
    # unified coordinate
    rotate_v = np.array([-np.pi / 2, 0, 0])
    rotate_m = cv2.Rodrigues(rotate_v)[0]
    # print('rotate_m',rotate_m)

    kp3ds_world_new = (rotate_m @ kp3ds_world.T).T


    camkp3d_world_new = (rotate_m @ camkp3d_world.T).T

    # project choosed vert to picture
    in_cam = np.array(
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2], camDists[0], camDists[1], camDists[4], camDists[2], camDists[3]])
    in_cam = np.expand_dims(in_cam, axis=0)
    # print('vert_min:',vert_min.shape) #(3,)
    # if want (X,0,Z)


    # way new 1
    choose_id = 0

    vert_choose = (kp3ds_world_new[45] + kp3ds_world_new[46] + kp3ds_world_new[16] + kp3ds_world_new[17]) / 4

    vert_min = vert_choose.copy()
    vert_min[1] = 0
    vert_world = (np.linalg.inv(rotate_m) @ vert_min.T).T
    vert_cam = world_to_camera(np.expand_dims(vert_world, axis=0), R, T)
    lock_2d = project_to_2d(np.expand_dims(vert_cam, axis=0), in_cam).numpy()
    lock_2d = np.squeeze(lock_2d, axis=0)
    # print('lock_2d:',lock_2d.shape) #(1,2)

    trans_end_world = (np.linalg.inv(rotate_m) @ vert_choose.T).T
    trans_end_cam = world_to_camera(np.expand_dims(trans_end_world, axis=0), R, T)
    trans_cam = trans_end_cam - vert_cam  # trans_start = vert_choose trans_end = vert_min
    trans_cam_d = np.linalg.norm(trans_cam)


    trans_need = trans_cam - (kp3ds[0, 45] + kp3ds[0, 46] + kp3ds[0, 16] + kp3ds[0, 17]) / 4
    trans_need_d = np.linalg.norm(trans_need)

    # SMPL GT
    kp3ds_pro_from_smpl = project_to_2d(np.expand_dims(kp3ds_new_cam, axis=0), in_cam).numpy()
    # kp3ds_pro_from_cam = project_to_2d(kp3ds,in_cam).numpy()

    # ground
    point_unified = np.array([[0, 0, 0], [100, 0, 0], [0, 0, 100]])
    point_world = (np.linalg.inv(rotate_m) @ point_unified.T).T
    point_cam = world_to_camera(point_world, R, T)
    ground_cam = estimate(point_cam)

    lock_info = {'lock_2d': lock_2d, 'lock_3d': vert_cam,  'smpl_2d': kp3ds_pro_from_smpl, \
                 'ground': ground_cam, 'trans_cam': trans_cam,
                 'trans_cam_d': trans_cam_d, \
                 'trans_need': trans_need, 'trans_need_d': trans_need_d}

    return lock_info


def get_lock_infos_pan_torso(R, T, camK, dist, kp3ds, rotate_m):
    # ground
    point_unified = np.array([[0, 0, 0], [100, 0, 0], [0, 0, -100]])
    # point_world = (np.linalg.inv(rotate_m) @ point_unified.T).T
    # print('point_world',point_world)
    point_world = point_unified
    point_cam = convert_wc_to_cc(point_world, R, T)
    ground_cam = estimate(point_cam)

    kp3ds_world = convert_cc_to_wc(kp3ds, R, T)
    # in_cam = np.array([camK[0,0],camK[1,1],camK[0,2],camK[1,2],dist[0],dist[1],dist[2],dist[3],dist[4]])
    in_cam = np.array([camK[0, 0], camK[1, 1], camK[0, 2], camK[1, 2], 0, 0, 0, 0, 0])
    in_cam = np.expand_dims(in_cam, axis=0)
    lock_2ds = []
    trans_cams = []
    trans_cams_d = []
    trans_needs = []
    trans_needs_d = []
    for i, kp3d in enumerate(kp3ds_world):
        kp3d_world_new = (rotate_m @ kp3d.T).T
        vert_choose = (kp3d_world_new[45] + kp3d_world_new[46] + kp3d_world_new[16] + kp3d_world_new[17]) / 4

        vert_min = vert_choose.copy()
        vert_min[1] = 0
        vert_world = (np.linalg.inv(rotate_m) @ vert_min.T).T
        vert_cam = convert_wc_to_cc(np.expand_dims(vert_world, axis=0), R, T)
        lock_2d = project_to_2d(np.expand_dims(vert_cam, axis=0), in_cam).numpy()
        lock_2d = np.squeeze(lock_2d, axis=0)

        trans_end_world = (np.linalg.inv(rotate_m) @ vert_choose.T).T
        trans_end_cam = convert_wc_to_cc(np.expand_dims(trans_end_world, axis=0), R, T)
        trans_cam = trans_end_cam - vert_cam  # trans_start = vert_choose trans_end = vert_min
        trans_cam_d = np.linalg.norm(trans_cam)
        # trans_need = trans_cam - verts[0,choose_id]
        root_inds = [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]
        root_trans = kp3ds[i][root_inds].mean(0)[None]
        camkp3d = kp3ds[i]
        trans_need = root_trans + trans_cam - (camkp3d[45] + camkp3d[46] + camkp3d[16] + camkp3d[17]) / 4
        trans_need_d = np.linalg.norm(trans_need)

        # for test
        # lock 2d to 3d cam
        lock_2d_qici = np.ones((len(lock_2d), 3))
        lock_2d_qici[:, :2] = lock_2d[:, :2]
        fx2_reci = 1 / camK[0, 0]
        fy2_reci = 1 / camK[1, 1]
        new_depthb = - ground_cam[3] / (
                    ground_cam[0] * fx2_reci * (lock_2d_qici[0, 0] - camK[0, 2]) + ground_cam[1] * fy2_reci * (
                        lock_2d_qici[0, 1] - camK[1, 2]) + ground_cam[2])
        new_Xb = np.matmul(np.linalg.inv(camK), (lock_2d_qici.T) * new_depthb)
        lock_3d = new_Xb.squeeze()

        lock_2ds.append(lock_2d)
        trans_cams.append(trans_cam)
        trans_cams_d.append(trans_cam_d)
        trans_needs.append(trans_need)
        trans_needs_d.append(trans_need_d)

    lock_2ds = np.array(lock_2ds)
    trans_cams = np.array(trans_cams)
    trans_cams_d = np.array(trans_cams_d)
    trans_needs = np.array(trans_needs)
    trans_needs_d = np.array(trans_needs_d)

    # ground
    # point_unified = np.array([[0,0,0],[100,0,0],[0,0,100]])
    # point_world = (np.linalg.inv(rotate_m) @ point_unified.T).T
    # point_cam = convert_wc_to_cc(point_world,R, T)
    # ground_cam = estimate(point_cam)

    lock_info = {'lock_2d': lock_2ds, 'ground': ground_cam, 'trans_cam': trans_cams, 'trans_cam_d': trans_cams_d, \
                 'trans_need': trans_needs, 'trans_need_d': trans_needs_d}
    return lock_info


def get_lock_infos_pan_way_2(R,T,camK,kp3ds,rotate_m):

    #ground
    point_unified = np.array([[0,0,0],[100,0,0],[0,0,-100]])
    # point_world = (np.linalg.inv(rotate_m) @ point_unified.T).T
    # print('point_world',point_world)
    point_world = point_unified
    point_cam = convert_wc_to_cc(point_world,R, T)
    ground_cam = estimate(point_cam)

    kp3ds_world = convert_cc_to_wc(kp3ds,R,T)
    # in_cam = np.array([camK[0,0],camK[1,1],camK[0,2],camK[1,2],dist[0],dist[1],dist[2],dist[3],dist[4]])
    in_cam = np.array([camK[0,0],camK[1,1],camK[0,2],camK[1,2],0,0,0,0,0])
    in_cam = np.expand_dims(in_cam,axis=0)
    lock_2ds = []
    lock_3ds = []
    trans_needs = []
    
    for i,kp3d in enumerate(kp3ds_world) :
        kp3d_world_new = (rotate_m @ kp3d.T).T
        vert_choose = (kp3d_world_new[45] + kp3d_world_new[46] + kp3d_world_new[16] + kp3d_world_new[17]) / 4
        
        vert_min = vert_choose.copy()
        vert_min[1] = 0
        vert_world = (np.linalg.inv(rotate_m) @ vert_min.T).T
        vert_cam = convert_wc_to_cc(np.expand_dims(vert_world,axis=0),R, T)
        lock_2d = project_to_2d(np.expand_dims(vert_cam,axis=0),in_cam).numpy()
        lock_2d = np.squeeze(lock_2d,axis=0)

        
        # print('root_trans',root_trans)
        ankle_center = (kp3ds[i][constants.SMPL_ALL_54['R_Ankle']] + kp3ds[i][constants.SMPL_ALL_54['L_Ankle']]) /2
        # print('ankle_center',ankle_center)
        trans_need = ankle_center - vert_cam 
        
        
        lock_2ds.append(lock_2d)
        lock_3ds.append(vert_cam)
        trans_needs.append(trans_need)
        

    lock_2ds = np.array(lock_2ds)
    lock_3ds = np.array(lock_3ds)
    trans_needs = np.array(trans_needs)
   
    lock_info = {'lock_2d':lock_2ds,'lock_3d':lock_3ds,'ground':ground_cam,\
        'trans_need':trans_needs}
    return lock_info

def get_lock_infos_way_2(info,R_t,K,kp3ds):
    #align with camkp3d
    camkp3d = info['kp3d_mono'].reshape(-1,3).copy()  #(32,3)
    mapper_32_to_torso = constants.joint_mapping(constants.H36M_32,constants.torso)
    camkp3d_32_to_torso = map_kps(camkp3d, maps=mapper_32_to_torso)
    mapper_54_to_torso = constants.joint_mapping(constants.SMPL_ALL_54,constants.torso)
    kp3ds_54_to_32 = map_kps(kp3ds.squeeze(), maps=mapper_54_to_torso)
    trans_T = (camkp3d_32_to_torso - kp3ds_54_to_32).mean(axis=0)
    # trans_T = camkp3d[3,:] - kp3ds[0,8,:]
    
    kp3ds_new_cam = kp3ds[0,:] + trans_T
    # print('verts_new_cam:',verts_new_cam.shape)  #(6890,3)
    #convert to world coordinate
    # R_t = h36m_cameras_extrinsic_params[subject_id][cam_view_id]
    R = np.array(R_t['orientation'])
    T = np.array(R_t['translation']) /1000
    kp3ds_org = kp3ds_new_cam.copy()
    kp3ds_float = np.squeeze(kp3ds_org)
    kp3ds_world = camera_to_world(kp3ds_float,R, T)
    camkp3d_world = camera_to_world(camkp3d,R,T)    
    #unified coordinate
    rotate_v=np.array([-np.pi/2, 0, 0])
    rotate_m=cv2.Rodrigues(rotate_v)[0]
    # print('rotate_m',rotate_m)
    kp3ds_world_new = (rotate_m @ kp3ds_world.T).T
    # print('verts_world_new:',verts_world_new.shape)  #(6890,3)
    # print('kp3ds_world_new :',kp3ds_world_new .shape)  #(54,3)
    camkp3d_world_new = (rotate_m @ camkp3d_world.T).T    
    
    #project choosed vert to picture
    # in_cam = np.array([K[0,0],K[1,1],K[0,2],K[1,2],camDists[0],camDists[1],camDists[4],camDists[2],camDists[3]])
    in_cam = np.array([K[0,0],K[1,1],K[0,2],K[1,2],0,0,0,0,0])
    in_cam = np.expand_dims(in_cam,axis=0)
    # print('vert_min:',vert_min.shape) #(3,)
   
    vert_choose = (kp3ds_world_new[constants.SMPL_ALL_54['R_Hip']] + kp3ds_world_new[constants.SMPL_ALL_54['L_Hip']] \
        + kp3ds_world_new[constants.SMPL_ALL_54['L_Shoulder']] + kp3ds_world_new[constants.SMPL_ALL_54['R_Shoulder']]) / 4
    

    vert_min = vert_choose.copy()
    vert_min[1] = 0
    vert_world = (np.linalg.inv(rotate_m) @ vert_min.T).T
    vert_cam = world_to_camera(np.expand_dims(vert_world,axis=0),R, T)
    lock_2d = project_to_2d(np.expand_dims(vert_cam,axis=0),in_cam).numpy()
    lock_2d = np.squeeze(lock_2d,axis=0)
    # print('lock_2d:',lock_2d.shape) #(1,2)

    ankle_center = (camkp3d[constants.H36M_32['R_Ankle']] + camkp3d[constants.H36M_32['L_Ankle']]) /2
    trans_need =  ankle_center - vert_cam
    
    
    #ground
    point_unified = np.array([[0,0,0],[100,0,0],[0,0,100]])
    point_world = (np.linalg.inv(rotate_m) @ point_unified.T).T
    point_cam = world_to_camera(point_world,R, T)
    ground_cam = estimate(point_cam)

    lock_info = {'lock_2d':lock_2d,'lock_3d':vert_cam,\
        'ground':ground_cam,'trans_need':trans_need}
    return lock_info

def dist_point_ground(ground, points):
    '''
    ground: m,4
    points: m, 3
    '''
    # fenzi m,N         fenmu  m, 1
    fenzi = ground[0] * points[:, 0] + ground[1]* points[:, 1] + ground[2] * points[:, 2] + ground[3]
    fenmu = np.sqrt(ground[0]**2 + ground[1]**2 +ground[2]**2)
    dist = fenzi / fenmu
    return dist  #m, N
