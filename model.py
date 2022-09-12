import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, RotateAxisAngle
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

import losses

class PoseOptimizer(nn.Module):
    def __init__(self, num_frames, num_hypos, renderer, rasterizer, 
                    smpl_verts, smpl_faces, scale, focal_length, img_w, img_h,
                    num_parts=2, angles=None, translation=None, scales=None, smpl_offset=None, align_idx=0,
                    mov_idx=0, align_axis=[[0., 0., 0.]], mov_scale=1.0,
                    obj_path='data/cuboid.obj', device='cuda'):
        super().__init__()
        self.num_frames = num_frames
        self.num_hypos = num_hypos
        self.num_parts = num_parts
        self.register_buffer('align_idx', torch.tensor(align_idx))
        self.register_buffer('align_axis', torch.tensor(align_axis))
        self.register_buffer('mov_idx', torch.tensor(mov_idx))

        verts, faces, _ = load_obj(obj_path)
        num_verts = verts.shape[0]

        verts = verts.unsqueeze(0).repeat(num_frames * num_hypos * num_parts, 1, 1)

        # modifiying faces to correspond to part vertices
        faces = faces.verts_idx
        faces = torch.cat([faces + i * num_verts for i in range(self.num_parts)], dim=0)
        faces = faces.unsqueeze(0).repeat(num_frames * num_hypos, 1, 1)

        self.register_buffer('verts', verts)
        self.register_buffer('faces', faces)

        verts_rgb = torch.ones_like(self.verts) # (B, V, 3)
        self.register_buffer('verts_rgb', verts_rgb)
        
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=device)
        self.T[0,2] = 0.0 
        self.renderer = renderer
        self.rasterizer = rasterizer
        
        # Create an optimizable parameter for the rotation and translation matrices
        if angles is None:
            angles = matrix_to_rotation_6d(torch.eye(3).unsqueeze(0))
            self.static_angles = nn.Parameter(angles)
        else:
            self.register_buffer('static_angles', angles.detach())

        if translation is None:
            translation = torch.rand(num_hypos, 1, 3) * 0.1
            translation[..., 2] += 2.5
            self.static_translation = nn.Parameter(translation)

        else:
            self.register_buffer('static_translation', translation)

        rot_angles = torch.arange(0, 1, 1./self.num_frames)
        self.rot_angles = nn.Parameter(rot_angles)

        if scales is None:
            scales = torch.ones(num_hypos * num_parts, 1, 3)
            self.base_scale = nn.Parameter(scales[0:1])
            self.move_scale = nn.Parameter(scales[1:2] * mov_scale)
        self.scale_thresh = 0.0001

        # smpl parameters
        self.register_buffer('smpl_verts', smpl_verts)
        self.register_buffer('smpl_faces', smpl_faces)

        if smpl_offset is None:
            smpl_offset = np.zeros((self.num_frames, 3), dtype=np.float32)
            smpl_offset[:,0] = 0.0
            smpl_offset[:,1] = 0.0
            smpl_offset[:,2] = 2.5
            smpl_offset = torch.from_numpy(smpl_offset)
            self.smpl_offset = nn.Parameter(smpl_offset)
        else:
            self.register_buffer('smpl_offset', smpl_offset)

        smplmesh_calibrate_path = 'data/smplmesh-calibrate.obj'
        smplmesh_calibrate =  o3d.io.read_triangle_mesh(smplmesh_calibrate_path)
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
        
        self.register_buffer('K', K.float())
        self.normalize = 1.0/(0.5*(self.img_h+self.img_w))

        # curve rotation in 3D
        yaw_degree2 = 0.0 * 180/np.pi
        rot_mat2 = RotateAxisAngle(yaw_degree2, axis='Y').get_matrix()
        rot_mat2 = rot_mat2[0,:3,:3].unsqueeze(0)
        ortho6d2 = matrix_to_rotation_6d(rot_mat2)
        self.curve_rot_angle = nn.Parameter(ortho6d2)
        curve_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.curve_offset = nn.Parameter(torch.from_numpy(curve_offset))

        # initialize intersection penalty
        self.intersection_loss = losses.IntersectionPenalty()

        self.infer_cache = None
        self.t_verts_cache = None
        self.smpl_verts_cache = None

    @property
    def scales(self):
        return torch.cat([self.base_scale, self.move_scale])

    @property
    def scales_clamped(self):
        base_scale = self.base_scale.clone()
        base_scale[base_scale > 0.] = base_scale[base_scale > 0.].clamp(min=self.scale_thresh)
        base_scale[base_scale < 0.] = base_scale[base_scale < 0.].clamp(max=-self.scale_thresh)

        move_scale = self.move_scale.clone()
        move_scale[move_scale > 0.] = move_scale[move_scale > 0.].clamp(min=self.scale_thresh)
        move_scale[move_scale < 0.] = move_scale[move_scale < 0.].clamp(max=-self.scale_thresh)

        return torch.cat([base_scale, move_scale])

    def render_verts(self, verts, faces):
        # rendering for silhouette loss
        textures = TexturesVertex(verts_features=torch.ones_like(verts))
        mesh = Meshes(verts, faces, textures)

        sil, frag = self.renderer(mesh, R=self.R, T=self.T)
        return sil, frag.zbuf[:, :, :, 0]

    def _get_transformed_verts(self):

        smpl_verts = self.smpl_verts.clone()
        smpl_verts *= self.h_ratio

        smpl_offset = self.smpl_offset.reshape(-1,1,3).repeat(1,self.smpl_verts.shape[1],1)
        smpl_verts[:,:,0] += self.hscale*smpl_offset[:,:,0]
        smpl_verts[:,:,1] += self.hscale*smpl_offset[:,:,1]
        smpl_verts[:,:,2] += self.hscale*smpl_offset[:,:,2] 

        static_R = rotation_6d_to_matrix(self.static_angles).unsqueeze(0).repeat(self.num_frames, 1, 1, 1).view(-1, 3, 3)

        # get the rotation matrix for moving part
        _cubeaxis = self.align_axis.repeat(self.num_frames, 1, 1).transpose(1, 2).to(self.verts.device)
        axis = torch.bmm(static_R,  _cubeaxis)
        moving_R = axis_angle_to_matrix((axis * self.rot_angles[:, None, None]).squeeze(2))

        static_T = self.static_translation.unsqueeze(0).repeat(self.num_frames, 1, 1, 1).view(-1, 1, 3)

        S = self.scales_clamped
        S = S.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        verts = self.verts.view(self.num_frames * self.num_hypos, self.num_parts, 8, 3).clone() * S
        verts[:, 0] = verts[:, 0] - verts[:, 0, :1]
        verts[:, 1] = verts[:, 1] - verts[:, 1, self.mov_idx:self.mov_idx+1] + verts[:, 0, self.align_idx:self.align_idx+1]

        base_p = verts[:, 0].transpose(1, 2)
        base_p = torch.bmm(static_R, base_p).transpose(1, 2) + static_T

        move_p = verts[:, 1].transpose(1, 2)
        move_p = torch.bmm(static_R, move_p).transpose(1, 2) + static_T
        orig_p = move_p[:, self.mov_idx:self.mov_idx+1]
        move_p = torch.bmm(moving_R, (move_p - orig_p).transpose(1, 2)).transpose(1, 2) + orig_p

        verts = torch.cat([base_p.unsqueeze(1), move_p.unsqueeze(1)], dim=1)
        verts = verts.view(self.num_frames * self.num_hypos, -1, 3)

        return verts, smpl_verts

    def forward(self, batch):
        
        verts, smpl_verts = self._get_transformed_verts()

        K_batch = self.K.expand(self.smpl_verts.shape[0],-1,-1)

        # Prespective projection
        points_out_v = torch.bmm(smpl_verts, K_batch.permute(0,2,1))
        smpl_2d = points_out_v[...,:2] / points_out_v[...,2:]

        # Human fitting error 
        l_points = torch.mean(self.normalize*(batch['points'] - smpl_2d)**2)

        # avg depth loss
        avg_depth_loss = F.relu(torch.abs(torch.mean(verts[:, :, 2], dim=1) - torch.mean(smpl_verts[:, :, 2].detach(), dim=1)) - 0.1)

        # # Hand & object 3D contact error 
        curve_rot_angle_mat = rotation_6d_to_matrix(self.curve_rot_angle)[0]
        care_idx = batch['care_idx']
        handcontact_v = batch['handcontact_v']

        obj_contact_curve = verts[:, 8:].mean(dim=1)[care_idx]
        smpl_contact_curve = smpl_verts[care_idx, handcontact_v, :].clone().detach()
        
        obj_contact_curve_after = torch.t(torch.mm(curve_rot_angle_mat, torch.t(obj_contact_curve))) + self.curve_offset
        contact_curve_loss = ((obj_contact_curve_after- smpl_contact_curve)**2).sum(1)

        if self.num_parts > 1:
            # intersection penalty
            verts_norm = verts.clone()
            inter_pen = self.intersection_loss(verts_norm, self.faces)

        else:
            inter_pen = torch.zeros(1).to(self.verts_rgb.device)

        # rendering for silhouette loss
        verts[:, :, 1:] *= -1
        textures = TexturesVertex(verts_features=self.verts_rgb.view(self.num_frames * self.num_hypos, -1, 3))
        mesh = Meshes(verts, self.faces, textures)

        silhouette, _ = self.renderer(meshes_world=mesh, R=self.R, T=self.T)
        _, h, w, c = silhouette.shape

        silhouette = silhouette.view(self.num_frames, self.num_hypos, h, w, c)
        
        # Calculate the silhouette loss
        sil_loss, _ = losses.silhouette_loss(silhouette[..., 3], batch['mask_ref'].unsqueeze(1)) #torch.mean((image[..., 3] - batch['mask_ref'].unsqueeze(1)) ** 2, dim=(2, 3)).view(-1)
        sil_dice = losses.dice_loss(silhouette[:, 0, :, :, 3], batch['mask_ref'])

        # Calculate parts silhoette losses
        p1_sil_loss = None
        p2_sil_loss = None
        verts_parts = verts.view(self.num_frames * self.num_hypos, self.num_parts, -1, 3)

        if 'p1_masks_ref' in batch:
            p1_sil, p1_depth = self.render_verts(verts_parts[:, 0], self.faces[:, :12])
            p1_sil = p1_sil.view(self.num_frames, self.num_hypos, h, w, c)

            p1_sil_loss, _ = losses.silhouette_loss(p1_sil[:, :, :, :, 3], batch['p1_masks_ref'].unsqueeze(1))
            p1_dice_loss = losses.dice_loss(p1_sil[:, 0, :, :, 3], batch['p1_masks_ref'])

        if 'p2_masks_ref' in batch:
            idx = 1 if 'p1_masks_ref' in batch else 0
            p2_sil, p2_depth = self.render_verts(verts_parts[:, idx], self.faces[:, :12])
            p2_sil = p2_sil.view(self.num_frames, self.num_hypos, h, w, c)

            p1_depth = p1_depth.view(self.num_frames, self.num_hypos, h, w)
            p2_depth = p2_depth.view(self.num_frames, self.num_hypos, h, w)
            p1_depth = torch.where(p1_depth == -1, torch.ones_like(p1_depth) * 1000, p1_depth)
            p2_depth = torch.where(p2_depth == -1, torch.ones_like(p2_depth) * 1000, p2_depth)
            d_mask = p2_depth < p1_depth

            p2_sil_loss, _ = losses.silhouette_loss(p2_sil[:, :, :, :, 3] * d_mask, batch['p2_masks_ref'].unsqueeze(1))
            p2_dice_loss = losses.dice_loss(p2_sil[:, 0, :, :, 3] * d_mask[:, 0], batch['p2_masks_ref'])


        d_mask = torch.sigmoid(p2_depth - p1_depth) * batch['p2_masks_ref'].unsqueeze(1)
        d_loss, _ = losses.silhouette_loss(d_mask, batch['p2_masks_ref'].unsqueeze(1))
        d_loss += losses.dice_loss(d_mask[:, 0], batch['p2_masks_ref'])

        # scale loss
        sc_loss = torch.sum(F.relu(0.001 - torch.abs(self.scales)))

        loss_dict = {}
        loss_dict['sil_loss'] = sil_loss
        loss_dict['p1_loss'] = p1_sil_loss if p1_sil_loss is not None else torch.zeros(1).to(self.verts_rgb.device)
        loss_dict['p2_loss'] = p2_sil_loss if p2_sil_loss is not None else torch.zeros(1).to(self.verts_rgb.device)
        loss_dict['sil_dice'] = sil_dice
        loss_dict['p1_dice'] = p1_dice_loss
        loss_dict['p2_dice'] = p2_dice_loss
        loss_dict['inter_pen'] = inter_pen
        loss_dict['l_points'] = l_points.mean()
        loss_dict['avg_depth_loss'] = avg_depth_loss
        loss_dict['contact_curve_loss'] = contact_curve_loss
        loss_dict['sc_loss'] = sc_loss
        loss_dict['part_dloss'] = d_loss
        return loss_dict

    def _transform_and_cache(self):

        t_verts, smpl_verts = self._get_transformed_verts()
        t_verts = t_verts.view(self.num_frames, self.num_hypos, -1, 3)

        smpl_verts = self.smpl_verts.clone()
        smpl_verts *= self.h_ratio

        smpl_offset = self.smpl_offset.reshape(-1,1,3).repeat(1,self.smpl_verts.shape[1],1) # (bs, 6890, 3)
        smpl_verts[:,:,0] += self.hscale*smpl_offset[:,:,0]
        smpl_verts[:,:,1] += self.hscale*smpl_offset[:,:,1]
        smpl_verts[:,:,2] += self.hscale*smpl_offset[:,:,2]

        self.t_verts_cache = t_verts
        self.smpl_verts_cache = smpl_verts

    def transform(self, idx=None):

        if self.t_verts_cache is None or self.smpl_verts_cache is None:
            self._transform_and_cache()

        if idx is None:
            return self.t_verts_cache, self.smpl_verts_cache 
        
        return self.t_verts_cache[idx], self.smpl_verts_cache[idx]

    def render(self, idx, batch, _cache=False):

        if (_cache and self.infer_cache is not None) or self.infer_cache is None:
            self.infer_cache = self.forward(batch)

        t_verts, smpl_verts = self.transform(idx)

        color = torch.ones_like(t_verts[0])[None]
        color[:, :8] = color[:, :8] * torch.tensor([[[0., 1., 0.]]], device=smpl_verts.device)
        color[:, 8:] = color[:, 8:] * torch.tensor([[[0., 0., 1.]]], device=smpl_verts.device)

        obj_text = TexturesVertex(verts_features=color)
        t_verts[:, :, 1:] *= -1
        obj_mesh = Meshes([t_verts[0]], [self.faces[idx]], obj_text)

        smpl_text = TexturesVertex(verts_features=torch.ones_like(smpl_verts)[None] * torch.tensor([[[1., 0., 0.]]], device=smpl_verts.device))
        smpl_verts[:, 1:] *= -1
        smpl_mesh = Meshes([smpl_verts], [self.smpl_faces[idx]], smpl_text)

        losses_idx = {}
        for k,v in self.infer_cache.items():
            if k in ['l_points', 'contact_curve_loss', 'avg_depth_loss', 'sc_loss']:
                losses_idx[k] = v.mean()
            elif ('reg' not in k):
                losses_idx[k] = v[idx]  
            else:
                losses_idx[k] = v[min(idx, v.shape[0]-1)]

        return obj_mesh, smpl_mesh, losses_idx

    def silhouette(self):
        verts = self.transform()
        mesh = Meshes(verts, self.faces, self.textures)

        return self.renderer(meshes_world=mesh, R=self.R, T=self.T)[0]

    @property
    def rotations(self):
        return rotation_6d_to_matrix(self.angles) 
