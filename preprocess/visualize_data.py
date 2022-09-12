

import random
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import os, re, glob, natsort, json

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    BlendParams,
    SoftSilhouetteShader,
    TexturesVertex
)

import sys
sys.path.insert(0, '../')
import utils

from natsort import natsorted
import imageio

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def set_seed(seed=2021):
    if seed < 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b

def merge_meshes(obj_path, device):
    """ helper function for loading and merging meshes. """
    verts_list = torch.empty(0,3)
    faces_list = torch.empty(0,3).long()
    num_vtx = [0]
    num_faces = [0]

    # merge meshes, load in ascending order
    meshes = natsort.natsorted(glob.glob(obj_path+'/final/*_rescaled_sapien.obj'))

    for part_mesh in meshes:
        #print('loading %s' %part_mesh)
        mesh = o3d.io.read_triangle_mesh(part_mesh)
        verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long()
        faces = faces + verts_list.shape[0]
        verts_list = torch.cat([verts_list, verts])
        faces_list = torch.cat([faces_list, faces])
        num_vtx.append(verts_list.shape[0])
        num_faces.append(faces_list.shape[0])

    verts_list = verts_list.to(device)
    faces_list = faces_list.to(device)

    return verts_list, faces_list, num_vtx, num_faces

def initialize_render(device, focal_x, focal_y, img_square_size):
    """ initialize camera, rasterizer, and shader. """
    # Initialize an OpenGL perspective camera.
    #cameras = FoVPerspectiveCameras(znear=1.0, zfar=9000.0, fov=20, device=device)
    #cameras = FoVPerspectiveCameras(device=device)
    #cam_proj_mat = cameras.get_projection_transform()
    img_square_center = int(img_square_size/2)

    camera_sfm = PerspectiveCameras(
                focal_length=((focal_x, focal_y),),
                principal_point=((img_square_center, img_square_center),),
                image_size = ((img_square_size, img_square_size),),
                in_ndc=False,
                device=device)

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_square_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    #lights = DirectionalLights(device=device, direction=((0, 0, 1),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=camera_sfm, lights=lights)
    )

    return phong_renderer

def load_motion(motions, device):
    """ load rotation axis, origin, and limit. """
    rot_origin = []
    rot_axis = []
    rot_type = []
    limit_a = []
    limit_b = []
    contact_list = []

    # load all meta data
    for idx, key in enumerate(motions.keys()):
        jointData = motions[key]

        # if contains movable parts
        if jointData is not None:
            origin = torch.FloatTensor(jointData['axis']['origin']).to(device)
            axis = torch.FloatTensor(jointData['axis']['direction']).to(device)
            mobility_type = jointData['type']
            if 'contact' in jointData:
                contact_list.append(jointData['contact'])

            # convert to radians if necessary
            if mobility_type == 'revolute':
                mobility_a = math.pi*jointData['limit']['a'] / 180.0
                mobility_b = math.pi*jointData['limit']['b'] / 180.0
            else:
                assert mobility_type == 'prismatic'
                mobility_a = jointData['limit']['a']
                mobility_b = jointData['limit']['b']

            rot_origin.append(origin)
            rot_axis.append(axis)
            rot_type.append(mobility_type)
            limit_a.append(mobility_a)
            limit_b.append(mobility_b)

    return rot_origin, rot_axis, rot_type, limit_a, limit_b, contact_list


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

def read_data(data_folder):
    """
    Load all annotated data for visualization
    """
    # load gt part motion values (degree or cm)
    gt_partmotion = []
    fp = open(os.path.join(data_folder, 'jointstate.txt'))
    for i, line in enumerate(fp):
        line = line.strip('\n')
        if isfloat(line) or isint(line):
            gt_partmotion.append(float(line))
    gt_partmotion = np.asarray(gt_partmotion)
           
    with open(os.path.join(data_folder, '3d_info.txt')) as myfile:
        gt_data = [next(myfile).strip('\n') for x in range(14)]
    
    # GT global object rotation 
    gt_pitch = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[3])[0])
    gt_yaw =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[4])[0])
    gt_roll = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[5])[0])

    # GT global object translation (cm)
    gt_x = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[6])[0])
    gt_y =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[7])[0])
    gt_z = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[8])[0])

    # GT object dimension (cm)
    gt_xdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[0])
    gt_ydim =  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[1])
    gt_zdim = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[9])[2])

    gt_cad = re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[10])[0]
    gt_part = int(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[11])[0])

    gt_focalX = int(float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[-2])[0]))
    gt_focalY = int(float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[-1])[0]))

    assert gt_focalX == gt_focalY

    data = {'part_motion': gt_partmotion,
            'pitch': gt_pitch,
            'yaw': gt_yaw,
            'roll': gt_roll,
            'x_offset': gt_x,
            'y_offset': gt_y,
            'z_offset': gt_z,
            'obj_size': [gt_xdim, gt_ydim, gt_zdim],
            'cad': gt_cad,
            'part': gt_part,
            'focal': gt_focalX}

    return data 

def visualize_optimal_poses_humans_video(model, images, vis_renderer, right_renderer, step=1, out_path=None, skip_interactive=False):
    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
    for idx in range(0, images.shape[0], step):
        R, T = look_at_view_transform(0.1, 0.0, 0.0,device='cuda')
        T[0,2] = 0.0  # manually set to zero
        MESH = model.render(idx)

        projection = vis_renderer(MESH, R=R, T=T)
        # left - right viewpoint
        bbox = MESH.get_bounding_boxes()
        _at = bbox[0].mean(dim=1)
        R, T = look_at_view_transform(_at[-1], 0, 90, at=_at[None], device='cuda')
        left_proj = right_renderer(MESH, R=R, T=T)

        R, T = look_at_view_transform(_at[-1], 0, 270, at=_at[None], device='cuda')
        right_proj = right_renderer(MESH, R=R, T=T)
        
        proj_frame = projection[0,...,:3].detach().cpu().numpy()
        
        H, W, _ = images[idx].shape
        if H > W:
            diff = (H - W) // 2
            proj_frame = proj_frame[:, diff:-diff]
        else:
            diff = (W - H) // 2
            proj_frame = proj_frame[diff:-diff, :]

        left_frame = left_proj[0, ..., :3].detach().cpu().numpy()
        right_frame = right_proj[0, ..., :3].detach().cpu().numpy()

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(images[idx])
        ax.imshow(proj_frame, alpha=0.4 )
        ax.set_title("Overlayed")
        # ax.axis('off')

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(left_frame)
        ax.set_title("Left View Point")
        ax.axis('off')

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(right_frame)
        ax.set_title("Right View Point")
        ax.axis('off')

        if out_path is not None:
            plt.savefig(os.path.join(out_path, f'{idx:05d}.jpg'), bbox_inches='tight',dpi=100)
            plt.close(fig)
        if not skip_interactive:
            plt.show()

    if out_path is not None:
        ppaths = natsorted(glob.glob(os.path.join(out_path, '*.jpg')))
        _plots = []
        for pp in ppaths:
            _plots.append( imageio.imread(pp) )

        imageio.mimsave(os.path.join(out_path, 'final_result.gif'), _plots, duration=0.1)

class PoseOptimizer(nn.Module):
    def __init__(self, instance_id, num_frames, smpl_verts, smpl_faces, scale, focal_length, img_w, img_h,
                    obj_verts, obj_faces, obj_size, vertexSegs, rot_type, rot_o, axis, part_idx, part_motion, pitch, yaw, roll,
                    x_offset, y_offset, z_offset, device='cuda'):
        super().__init__()
        self.id = instance_id,
        self.num_frames = num_frames
        self.device = device
        self.obj_verts = obj_verts.detach()
        self.obj_faces = obj_faces.detach()
        self.rot_type = rot_type
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset 
        self.part_motion = part_motion 

        # smpl parameters
        self.register_buffer('smpl_verts', smpl_verts)
        # self.smpl_faces = smpl_faces
        self.register_buffer('smpl_faces', smpl_faces)

        smpl_offset = np.zeros((self.num_frames, 3), dtype=np.float32)
        smpl_offset[:,0] = 0.0
        smpl_offset[:,1] = 0.0
        smpl_offset[:,2] = 2.5
        self.smpl_offset = nn.Parameter(torch.from_numpy(smpl_offset))

        smplmesh_calibrate_path = '../data/smplmesh-calibrate.obj'
        smplmesh_calibrate =  o3d.io.read_triangle_mesh(smplmesh_calibrate_path) # load smpl mesh
        hverts_cal = torch.from_numpy(np.asarray(smplmesh_calibrate.vertices)).float()
        human_height = 175 #cm
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

        # camera is almost at the center (distance can't be zero for diff render)
        self.R, self.T = look_at_view_transform(0.1, 0.0, 0.0,device=self.device)
        self.T[0,2] = 0.0  # manually set to zero

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

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset

        # information purposes only
        self.human_height = human_height
        self.obj_size = obj_size
        self.object_part_idx = part_idx

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
        
        return smpl_verts

    def render(self, idx):

        smpl_verts = self.transform(idx)

        smpl_text = TexturesVertex(verts_features=torch.ones_like(smpl_verts)[None] * torch.tensor([[[1., 0., 0.]]], device=smpl_verts.device))
        smpl_verts[:, 1:] *= -1
        smpl_mesh = Meshes([smpl_verts], [self.smpl_faces[idx]], smpl_text)

        partmotion = self.part_motion[idx]
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

        # step 4: object offset
        obj_verts[:, 0] += 100.0*self.x_offset
        obj_verts[:, 1] += 100.0*self.y_offset
        obj_verts[:, 2] += 100.0*self.z_offset

        obj_verts[:,1:] *= -1
        # create object mesh for diff render and visualization
        tex = torch.ones_like(obj_verts).unsqueeze(0)
        tex[:, :, 0] = 0
        tex[:, :, 1] = 1
        tex[:, :, 2] = 0
        textures = TexturesVertex(verts_features=tex).to(self.device)
        obj_mesh = Meshes(verts=[obj_verts],faces=[self.obj_faces],textures=textures)

        MESH = join_meshes_as_scene([obj_mesh, smpl_mesh])

        return MESH

    def to_json(self):
        # output the optimized poses
        camera = {
            'focal_length': self.focal,
            'rotation': self.R.cpu().tolist(),
            'translation': self.T.cpu().tolist(),
            'image_width': self.img_w,
            'image_height': self.img_h
        }
        human = {
            'hscale': self.hscale,
            'hratio': self.h_ratio.cpu().tolist(),
            'height': self.human_height
        }
        obj = {
            'offset': [self.x_offset, self.y_offset, self.z_offset],
            'ratio': [self.x_ratio.cpu().tolist(), self.y_ratio.cpu().tolist(), self.z_ratio.cpu().tolist()],
            'obj_size': self.obj_size,
            'part_idx': self.object_part_idx,
            'part_motion': self.part_motion.tolist(),
            'rot_type': self.rot_type,
            'rot_origin': self.rot_o.cpu().tolist(),
            'axis': self.axis.cpu().tolist()
        }
        output = {
            'id': self.id,
            'camera': camera,
            'human': human,
            'object': obj,
            'num_frames': self.num_frames,
            'smpl_offset': self.smpl_offset.cpu().tolist()
        }
        return output

    def load_offsets_from_json(self, json):
        self.smpl_offset = nn.Parameter(torch.as_tensor(json["smpl_offset"], device=self.smpl_verts.device))




def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--cad_path', type=str, required=True, help="path to cad models")
    parser.add_argument('--load_optimized_path', type=str, help="path to which to save optimized results")
    parser.add_argument('--save_optimized_path', type=str, help="path to which to load optimized results")
    parser.add_argument('--save_images_path', type=str, help="path to which to save the images")
    parser.add_argument('--skip_interactive', action="store_true", help="whether to skip interactive display of images")
    parser.add_argument('--iterations', type=int, default=500, help="Number of iterations to run optimization")
    parser.add_argument('--seed', type=int, default=-1, help="seed for prngs")
    parser.add_argument('--step', type=int, default=1, help="step to sample frames while loading")
    parser.add_argument('--visualize_step', type=int, default=1, help="step to sample frames while visualizing")

    return parser.parse_args()

def find_optimal_pose(model, smpl_points, lr=0.01, start_it=0, iterations=500,):


    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.75 * iterations), gamma=0.1)

    batch = {
        'iter': 0,
        'points': torch.from_numpy(smpl_points).to(device),
    }

    loop = tqdm(range(start_it, start_it+iterations))
    for i in loop:

        batch['iter'] = i

        losses = model(batch)

        loss = losses['l_points'].mean()

        optimizer.zero_grad()
        # print('LOSSES: ', total_loss.shape)
        # total_loss.backward(torch.ones_like(total_loss))

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loop.set_description('Optimizing loss:  %.4f' % (loss.mean()))

    return model

def main():
    
    args = parse_args()

    # seeding
    set_seed(args.seed)

    IMAGE_PATH = os.path.join(args.data_path, 'frames')
    SMPL_MESH_PATH = os.path.join(args.data_path, 'smplmesh')
    SMPL_POINT_PATH = os.path.join(args.data_path, 'smplv2d')

    gt_data = read_data(args.data_path)
    
    x_offset = gt_data['x_offset']
    y_offset = gt_data['y_offset']
    z_offset = gt_data['z_offset']
    yaw = gt_data['yaw']
    pitch = gt_data['pitch']
    roll = gt_data['roll']
    part_motion = gt_data['part_motion']
    obj_size = gt_data['obj_size'] # length, height, width (x, y, z), cm
    focal_x = gt_data['focal']
    focal_y = gt_data['focal']

    images = utils.get_frames(IMAGE_PATH, args.step)
    N, H, W, _ = images.shape

    # loading objecy
    obj_path = os.path.join(args.cad_path, gt_data['cad'])
    verts, faces, vertexSegs, faceSegs = merge_meshes(obj_path, device)
    verts[:,1:] *= -1  # pytorch3d -> world coordinate
    obj_verts = verts.to(device)
    obj_faces = faces.to(device)

    # load motion json file
    with open(os.path.join(args.cad_path, gt_data['cad'], 'motion.json')) as json_file:
        motions = json.load(json_file)
    assert len(motions) + 2 == len(vertexSegs)
    rot_o, rot_axis, rot_type, limit_a, limit_b, contact_list = load_motion(motions, device)

    # loading smpls
    smpl_verts, smpl_faces = utils.get_smpl_meshes(SMPL_MESH_PATH, args.step)
    smpl_points = utils.get_smpl_points(SMPL_POINT_PATH, args.step)

    phong_renderer = initialize_render(device, focal_x=focal_x, focal_y=focal_y, 
                                                        img_square_size=max(H, W))

    # num_frames, smpl_verts, smpl_faces, scale, focal_length, img_w, img_h,
    #                 obj_verts, obj_size, vertexSegs, rot_o, axis, part_idx, pitch, yaw, roll
    instance_id = args.data_path
    instance_id_pieces = instance_id.strip('/').split('/')
    if len(instance_id_pieces) > 2:
        instance_id = '/'.join(instance_id_pieces[len(instance_id_pieces)-2:])
    hmodel = PoseOptimizer(instance_id=instance_id, num_frames=images.shape[0], smpl_verts=smpl_verts, smpl_faces=smpl_faces, scale=100., focal_length=gt_data['focal'], 
                                img_w=W, img_h=H, obj_verts=obj_verts, obj_faces=obj_faces, obj_size=obj_size, vertexSegs=vertexSegs, 
                                rot_type=rot_type, rot_o=rot_o, axis=rot_axis, part_idx=gt_data['part'], part_motion=part_motion, 
                                pitch=pitch, yaw=yaw, roll=roll, x_offset=x_offset, y_offset=y_offset,
                                z_offset=z_offset).to(device)
    if args.load_optimized_path is not None:
        # TODO: read from optimized
        with open(args.load_optimized_path, 'r') as infile:
            json_data = json.load(infile)
            hmodel.load_offsets_from_json(json_data)
    else:
        hmodel = find_optimal_pose(model=hmodel, smpl_points=smpl_points, iterations=args.iterations)
    if args.save_optimized_path is not None:
        # Save to optimized
        with open(args.save_optimized_path, 'w') as save_file:
            json.dump(hmodel.to_json(), save_file)

    imsize = max(H, W)
    cameras = PerspectiveCameras(
                            focal_length=((focal_x, focal_x),),
                            principal_point=((imsize // 2, imsize // 2),),
                            image_size = ((imsize, imsize),),
                            in_ndc=False,
                            device=device)
    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=imsize,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None,
    )

    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))

    right_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    visualize_optimal_poses_humans_video(hmodel, images, phong_renderer, right_renderer, args.visualize_step, args.save_images_path, args.skip_interactive)

if __name__ == '__main__':
    main()