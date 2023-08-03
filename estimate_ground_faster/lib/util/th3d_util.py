
from pytorch3d.structures.meshes import Meshes
import torch
import pytorch3d.structures

#pytorch
import torch
import torch.optim

#pytorch3d
import pytorch3d.ops.subdivide_meshes
import pytorch3d.structures
import pytorch3d.io
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops.knn import knn_points
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io.ply_io import _load_ply
from pytorch3d.renderer import (
    TexturesVertex
)
from iopath.common.file_io import PathManager

from lib.io.io_3d import check_verts_color_in_obj, load_verts_color_from_obj


def load_textured_meshes_as_pth3d(path:str, device:torch.device, flag_normalization:bool=False) -> (Meshes):
    if path[-4:] == '.ply' :
        verts, faces, color, ____ = _load_ply(path, path_manager=PathManager())

        if flag_normalization :
            mid_point = torch.zeros((3))
            #compute height
            vert_max, ____ = torch.max(verts, axis=0)
            vert_min, ____ = torch.min(verts, axis=0)
            height = vert_max[1] - vert_min[1]
            for i in range(3):
                mid_point[i] = (vert_max[i] + vert_min[i]) / 2
            verts = verts - mid_point
            verts = verts * 1.8 / height
        if color is not None :
            textures = TexturesVertex(verts_features=color[None].to(device))
        else :
            textures = None
        mesh = Meshes(
            verts=verts[None].to(device),
            faces=faces[None].to(device),
            textures=textures
        )

    elif path[-4:] == '.obj' or path[-4:] == '.OBJ' :
        mesh = load_objs_as_meshes([path])

        if mesh.textures is None :
            vert_color_type = check_verts_color_in_obj(path)
            
            if vert_color_type is None:
                verts_color = torch.ones_like(mesh.verts_list()[0]) * 0.65
            else :
                assert vert_color_type is float or vert_color_type is int
                verts_color = load_verts_color_from_obj(path, vert_color_type)
                verts_color = torch.from_numpy(verts_color).to(torch.float32).to(mesh.device)

            mesh = Meshes(
                verts=mesh.verts_list(),
                faces=mesh.faces_list(),
                textures=TexturesVertex([verts_color])
            )
    else :
        raise(NotImplementedError('Not support %s' % path[-4:]))

    if flag_normalization :
        mesh_normalization(mesh)
        
    mesh= mesh.to(device=device)
    return mesh


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