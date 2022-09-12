import numpy as np
import torch
import torch.nn as nn

import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, look_at_view_transform

class PoseOptimizer(nn.Module):
    def __init__(self, num_frames, renderer, rasterizer, 
                    smpl_verts, smpl_faces, scale, focal_length, img_w, img_h, device='cuda'):
        super().__init__()
        self.num_frames = num_frames
        
        # camera is almost at the center (distance can't be zero for diff render)
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=device)
        self.T[0,2] = 0.0  # manually set to zero
        self.renderer = renderer
        self.rasterizer = rasterizer
        
        # Create an optimizable parameter for the rotation and translation matrices

        # smpl parameters
        self.register_buffer('smpl_verts', smpl_verts)
        # self.smpl_faces = smpl_faces
        self.register_buffer('smpl_faces', smpl_faces)

        smpl_offset = np.zeros((self.num_frames, 3), dtype=np.float32)
        smpl_offset[:,0] = 0.0
        smpl_offset[:,1] = 0.0
        smpl_offset[:,2] = 2.5
        self.smpl_offset = nn.Parameter(torch.from_numpy(smpl_offset))

        smplmesh_calibrate_path = 'data/smplmesh-calibrate.obj'
        smplmesh_calibrate =  o3d.io.read_triangle_mesh(smplmesh_calibrate_path) # load smpl mesh
        hverts_cal = torch.from_numpy(np.asarray(smplmesh_calibrate.vertices)).float()
        human_height = 1.75 #m
        h_diff = torch.max(hverts_cal[:,1]) - torch.min(hverts_cal[:,1])
        self.h_ratio = (human_height / h_diff).detach()
        self.hscale = scale

        self.focal = focal_length
        self.img_w = img_w
        self.img_h = img_h

        K = torch.from_numpy(np.array([[self.focal, 0, self.img_w/2],
                                       [0, self.focal, self.img_h/2],
                                       [0, 0, 1]]))
        # self.K = K.float().to(self.device)
        self.register_buffer('K', K.float())
        self.normalize = 1.0/(0.5*(self.img_h+self.img_w))

    def render_verts(self, verts, faces):
        # rendering for silhouette loss
        textures = TexturesVertex(verts_features=torch.ones_like(verts))
        mesh = Meshes(verts, faces, textures)

        sil = self.renderer(mesh, R=self.R, T=self.T)
        return sil

    def forward(self, batch):
        
        smpl_verts = self.smpl_verts.clone()
        smpl_verts *= self.h_ratio

        smpl_offset = self.smpl_offset.reshape(-1,1,3).repeat(1,self.smpl_verts.shape[1],1) # (bs, 6890, 3)
        smpl_verts[:,:,0] += self.hscale*smpl_offset[:,:,0]
        smpl_verts[:,:,1] += self.hscale*smpl_offset[:,:,1]
        smpl_verts[:,:,2] += self.hscale*smpl_offset[:,:,2] 

        K_batch = self.K.expand(self.smpl_verts.shape[0],-1,-1)

        # Prespective projection
        points_out_v = torch.bmm(smpl_verts, K_batch.permute(0,2,1))
        # print('points_out_v:', points_out_v.shape, points_out_v[0].min(), points_out_v[0].max())
        smpl_2d = points_out_v[...,:2] / points_out_v[...,2:]

        # Human fitting error 
        l_points = torch.mean(self.normalize*(batch['points'] - smpl_2d)**2)

        loss_dict = {}
        loss_dict['l_points'] = l_points.mean()
        return loss_dict

    def transform(self, idx=None):

        if idx is None:

            smpl_verts = self.smpl_verts.clone()
            smpl_verts *= self.h_ratio

            smpl_offset = self.smpl_offset.reshape(-1,1,3).repeat(1,self.smpl_verts_orig.shape[1],1) # (bs, 6890, 3)
            smpl_verts[:,:,0] += self.hscale*smpl_offset[:,:,0]
            smpl_verts[:,:,1] += self.hscale*smpl_offset[:,:,1]
            smpl_verts[:,:,2] += self.hscale*smpl_offset[:,:,2] 
        else:

            smpl_verts = self.smpl_verts[idx].clone()
            smpl_verts *= self.h_ratio

            smpl_offset = self.smpl_offset.reshape(-1,1,3).repeat(1,self.smpl_verts.shape[1],1)[idx] # (bs, 6890, 3)
            smpl_verts[:,0] += self.hscale*smpl_offset[:,0]
            smpl_verts[:,1] += self.hscale*smpl_offset[:,1]
            smpl_verts[:,2] += self.hscale*smpl_offset[:,2] 
        
        return None, smpl_verts

    def render(self, idx):

        _, smpl_verts = self.transform(idx)

        smpl_text = TexturesVertex(verts_features=torch.ones_like(smpl_verts)[None] * torch.tensor([[[1., 0., 0.]]], device=smpl_verts.device))
        smpl_verts[:, 1:] *= -1
        smpl_mesh = Meshes([smpl_verts], [self.smpl_faces[idx]], smpl_text)

        return None, smpl_mesh

    def silhouette(self):
        verts = self.transform()
        mesh = Meshes(verts, self.faces, self.textures)

        return self.renderer(meshes_world=mesh, R=self.R, T=self.T)

    @property
    def rotations(self):
        return None
