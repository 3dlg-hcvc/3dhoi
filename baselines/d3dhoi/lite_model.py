import torch.nn as nn
import torch
from pytorch3d.renderer import (
    look_at_view_transform, euler_angles_to_matrix
)
from utils import rotation_matrix

class JOHMRLite(nn.Module):

    def __init__(self, x_offset, y_offset, z_offset, yaw, pitch, roll, part_motion, obj_size, \
                 obj_verts, obj_faces, part_idx, rot_o, axis, vertexSegs, faceSegs, rot_type):

        super().__init__()
        self.device = obj_verts.device
        self.obj_verts = obj_verts.detach()
        self.obj_faces = obj_faces.detach()
        self.rot_type = rot_type
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset 
        self.part_motion = part_motion 

        # camera is almost at the center (distance can't be zero for diff render)
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=self.device)
        self.T[0,2] = 0.0 

        obj_size = [x / 100. for x in obj_size]

        x_diff = torch.max(obj_verts[:,0]) - torch.min(obj_verts[:,0])
        self.x_ratio = float(obj_size[0]) / x_diff
        y_diff = torch.max(obj_verts[:,1]) - torch.min(obj_verts[:,1])
        self.y_ratio = float(obj_size[1]) / y_diff
        z_diff = torch.max(obj_verts[:,2]) - torch.min(obj_verts[:,2])
        self.z_ratio = float(obj_size[2]) / z_diff

        # predefined object CAD part and axis
        self.vertexStart = vertexSegs[part_idx]
        self.vertexEnd = vertexSegs[part_idx+1]
        self.faceStart = faceSegs[part_idx]
        self.faceEnd = faceSegs[part_idx+1]
        self.rot_o = rot_o[part_idx]
        self.axis = axis[part_idx]

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

        obj_verts[:,1:] *= -1

        part_verts = obj_verts[self.vertexStart:self.vertexEnd]
        part_faces = self.obj_faces[self.faceStart:self.faceEnd]

        return obj_verts, self.obj_faces, part_verts, part_faces, _rot_o, _axis