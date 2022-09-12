# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn
import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform, TexturesVertex
)
from pytorch3d.structures import Meshes

import os
from pytorch3d.transforms import (
    euler_angles_to_matrix
)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = torch.empty(3,3)

    rot_mat[0,0] = aa + bb - cc - dd
    rot_mat[0,1] = 2 * (bc + ad)
    rot_mat[0,2] = 2 * (bd - ac)

    rot_mat[1,0] = 2 * (bc - ad)
    rot_mat[1,1] = aa + cc - bb - dd
    rot_mat[1,2] = 2 * (cd + ab)

    rot_mat[2,0] = 2 * (bd + ac)
    rot_mat[2,1] = 2 * (cd - ab)
    rot_mat[2,2] = aa + dd - bb - cc

    return rot_mat

class JOHMRLite(nn.Module):

    def __init__(self, x_offset, y_offset, z_offset, yaw, pitch, roll, part_motion, obj_size, \
                 obj_verts, obj_faces, vis_render, part_idx, rot_o, axis, vertexSegs, faceSegs, rot_type):

        super().__init__()
        self.device = obj_verts.device
        self.vis_render = vis_render
        self.obj_verts = obj_verts.detach()
        self.obj_faces = obj_faces.detach()
        self.rot_type = rot_type
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset 
        self.part_motion = part_motion 
        
        # camera is almost at the center (distance can't be zero for diff render)
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=self.device)
        self.T[0,2] = 0.0  # manually set to zero

        obj_size = [x / 100. for x in obj_size]
        if self.rot_type[0] == 'prismatic':
            self.part_motion = self.part_motion / 100.

        x_diff = torch.max(obj_verts[:,0]) - torch.min(obj_verts[:,0])
        self.x_ratio = float(obj_size[0]) / x_diff
        y_diff = torch.max(obj_verts[:,1]) - torch.min(obj_verts[:,1])
        self.y_ratio = float(obj_size[1]) / y_diff
        z_diff = torch.max(obj_verts[:,2]) - torch.min(obj_verts[:,2])
        self.z_ratio = float(obj_size[2]) / z_diff

        # predefined object CAD part and axis
        self.vertexStart = vertexSegs[part_idx]
        self.vertexEnd = vertexSegs[part_idx+1]
        self.rot_o = rot_o[part_idx]
        self.axis = axis[part_idx]

        self.faceStart = faceSegs[part_idx]
        self.faceEnd = faceSegs[part_idx+1]

        # pytorch3d -> world coordinate
        self.rot_o[1:] *= -1
        self.axis[1:] *= -1

        # rescale object
        self.obj_verts[:, 0] *=  self.x_ratio
        self.obj_verts[:, 1] *=  self.y_ratio
        self.obj_verts[:, 2] *=  self.z_ratio
        self.rot_o[0] *= self.x_ratio
        self.rot_o[1] *= self.y_ratio
        self.rot_o[2] *= self.z_ratio

        euler_angle = torch.tensor([pitch, yaw, roll]).reshape(1,3)
        self.objR = euler_angles_to_matrix(euler_angle, ["X","Y","Z"]).to(self.device)[0]

        return



    def forward(self, index):

        partmotion = self.part_motion[index]
        obj_verts = self.obj_verts.clone()
        
        # part motion
        if self.rot_type[0] == 'prismatic':
            part_state = torch.tensor(partmotion).to(self.device)
            obj_verts_t1 = obj_verts[self.vertexStart:self.vertexEnd, :] - self.rot_o
            obj_verts_t2 = obj_verts_t1 + self.axis * part_state  #/float(annotation['obj_dim'][2]) * z_ratio
            obj_verts[self.vertexStart:self.vertexEnd, :] = obj_verts_t2 + self.rot_o

        else:
            part_state = torch.tensor(partmotion*0.0174533)
            part_rot_mat = rotation_matrix(self.axis, part_state)
            obj_verts_t1 = obj_verts[self.vertexStart:self.vertexEnd, :] - self.rot_o
            obj_verts_t2 = torch.mm(part_rot_mat.to(self.device), obj_verts_t1.permute(1,0)).permute(1,0)
            obj_verts[self.vertexStart:self.vertexEnd, :] = obj_verts_t2 + self.rot_o
              
        # step 3: object orientation
        obj_verts = torch.mm(self.objR, obj_verts.permute(1,0)).permute(1,0)

        _rot_o = torch.mm(self.objR, self.rot_o.unsqueeze(0).transpose(0, 1)).transpose(0, 1)
        _rot_o = _rot_o + torch.tensor([[self.x_offset, self.y_offset, self.z_offset]]).to(_rot_o.device)
        _axis = torch.mm(self.objR, self.axis.unsqueeze(0).transpose(0, 1)).transpose(0, 1)

        # step 4: object offset
        obj_verts[:, 0] += self.x_offset
        obj_verts[:, 1] += self.y_offset
        obj_verts[:, 2] += self.z_offset

        # print(obj_verts[:10])

        obj_verts[:,1:] *= -1

        part_verts = obj_verts[self.vertexStart:self.vertexEnd]
        part_faces = self.obj_faces[self.faceStart:self.faceEnd]
        part_faces = part_faces - part_faces.min()

        # create object mesh for diff render and visualization
        tex = torch.ones_like(obj_verts).unsqueeze(0)
        tex[:, :, 0] = 0
        tex[:, :, 1] = 1
        tex[:, :, 2] = 0
        textures = TexturesVertex(verts_features=tex).to(self.device)
        self.obj_mesh = Meshes(verts=[obj_verts],faces=[self.obj_faces],textures=textures)
        vis_image, frag = self.vis_render(meshes_world=self.obj_mesh, R=self.R, T=self.T)
        silhouette = vis_image[0,:,:,3]  

        # create part mesh for diff render and visualization
        tex = torch.ones_like(part_verts).unsqueeze(0)
        tex[:, :, 0] = 0
        tex[:, :, 1] = 1
        tex[:, :, 2] = 0
        textures = TexturesVertex(verts_features=tex).to(self.device)
        part_mesh = Meshes(verts=[part_verts],faces=[part_faces],textures=textures)
        part_image, _ = self.vis_render(meshes_world=part_mesh, R=self.R, T=self.T)
        silhouette_part = part_image[0,:,:,3] 

        return silhouette.detach().cpu().numpy(), frag.zbuf[..., 0].detach().cpu().numpy(), silhouette_part.detach().cpu().numpy(), obj_verts, self.obj_faces, part_verts, part_faces, _rot_o, _axis
