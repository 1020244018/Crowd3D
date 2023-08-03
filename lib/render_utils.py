import numpy as np
import pyrender, trimesh

def get_rotate_X(theta:float):
    rotate = np.array(
        [[1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]]
    )
    return rotate

def get_rotate_Y(theta:float):
    rotate = np.array(
        [[np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]]
    )
    return rotate

def get_rotate_Z(theta:float):
    rotate = np.array(
        [[np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]]
    )
    return rotate

def rotate_RT(RT:np.ndarray, rotate:np.ndarray):
    assert RT.shape == (4, 4)
    assert rotate.shape == (3, 3)
    
    new_RT = np.zeros((4, 4))
    new_RT[:, 0:4] = RT[:, 0:4]

    ori_R = RT[0:3, 0:3]
    new_R = np.matmul(rotate, ori_R)
    
    new_RT[0:3, 0:3] = new_R
    return new_RT

def trans_RT(RT:np.ndarray, trans:np.ndarray or list[float]):
    assert RT.shape == (4, 4)
    if type(trans) == np.ndarray:
        assert trans.shape == (3)
    elif type(trans) == list:
        assert len(trans) == 3
    else:
        raise(NotImplementedError(type(trans)))

    new_RT = RT.copy()
    for i in range(3):
        new_RT[i, 3] += trans[i]
    return new_RT

def render(list_mesh, f, cx, cy, camera_pose, light_pose, image_shape=None):
    scene = pyrender.Scene(bg_color=np.zeros(3))
    for mesh in list_mesh:
        mesh_py = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh_py)
    camera = pyrender.IntrinsicsCamera(fx=f, fy=f, cx=cx, cy=cy, znear=5, zfar=1000) # 5-1000
    scene.add(camera, pose=camera_pose)
    light = pyrender.light.DirectionalLight(color=np.ones((3)), intensity=8)
    scene.add(light, pose=light_pose)
    if image_shape is None:
        r = pyrender.OffscreenRenderer(cx*2, cy*2)
    else:
        r = pyrender.OffscreenRenderer(image_shape[1], image_shape[0])
    flag_render = pyrender.RenderFlags.SHADOWS_ALL | pyrender.RenderFlags.SKIP_CULL_FACES
    color, depth = r.render(scene, flags=flag_render)
    r.delete()
    return color, depth

def convert_trimesh(mesh_list, f):
    trimesh_list=[]
    for v in mesh_list:
        cur_trimesh=trimesh.Trimesh(vertices=v, faces=f)
        cur_trimesh.vertices[:, 1]*=-1
        cur_trimesh.vertices[:, 2]*=-1
        trimesh_list.append(cur_trimesh)

    return trimesh_list