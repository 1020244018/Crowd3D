
from pytorch3d.structures.meshes import Meshes
import torch
import pytorch3d.structures

def mesh_normalization(mesh:pytorch3d.structures.Meshes, TARGET_HEIGHT:float=1.8) :
    #push the mesh to (0, 0, 0)
    #old version
    #vert_avg = torch.mean(mesh.verts_list()[0], axis=0)
    #mesh._verts_list[0] = mesh.verts_list()[0] - vert_avg

    mid_point = torch.zeros((3)).to(mesh.device)

    #compute height
    vert_max, ____ = torch.max(mesh.verts_list()[0], axis=0)
    vert_min, ____ = torch.min(mesh.verts_list()[0], axis=0)
    height = vert_max[1] - vert_min[1]

    for i in range(3):
        mid_point[i] = (vert_max[i] + vert_min[i]) / 2

    mesh._verts_list[0] = mesh.verts_list()[0] - mid_point
    mesh._verts_list[0] = mesh._verts_list[0] * TARGET_HEIGHT / height

    return mesh

    #return Meshes(
    #    verts=[(mesh.verts_list()[0].clone().contiguous() - mid_point) * TARGET_HEIGHT / height],
    #    faces=[mesh.faces_list()[0].clone().contiguous()],
    #    textures=mesh.textures
    #)

def mesh_normalization_old(mesh:pytorch3d.structures.Meshes, TARGET_HEIGHT:float=1.8) :
    #push the mesh to (0, 0, 0)
    #old version
    vert_avg = torch.mean(mesh.verts_list()[0], axis=0)
    mesh._verts_list[0] = mesh.verts_list()[0] - vert_avg

    #compute height
    vert_max, ____ = torch.max(mesh.verts_list()[0], axis=0)
    vert_min, ____ = torch.min(mesh.verts_list()[0], axis=0)
    height = vert_max[1] - vert_min[1]
    mesh._verts_list[0] = mesh._verts_list[0] * TARGET_HEIGHT / height
    return

