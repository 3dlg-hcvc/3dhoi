import random
import argparse

import matplotlib.pyplot as plt
import os, re, glob
from natsort import natsorted

import math
import numpy as np
import torch
import imageio

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    TexturesVertex
)

import sys
sys.path.insert(0, '../') 
import utils

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

def initialize_render(device, focal_x, focal_y, img_square_size):
    """ initialize camera, rasterizer, and shader. """
    # Initialize an OpenGL perspective camera.
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

def visualize_optimal_poses_humans_video(models, images, vis_renderer, out_path=None, skip_interactive=False):


    for idx in range(images.shape[0]):
        R, T = look_at_view_transform(0.1, 0.0, 0.0,device='cuda')
        T[0,2] = 0.0  # manually set to zero
        MESH = models[idx]

        projection = vis_renderer(MESH, R=R, T=T)
        # left - right viewpoint
        bbox = MESH.get_bounding_boxes()
        _at = bbox[0].mean(dim=1)
        R, T = look_at_view_transform(_at[-1], 0, 90, at=_at[None], device='cuda')
        left_proj = vis_renderer(MESH, R=R, T=T)

        R, T = look_at_view_transform(_at[-1], 0, 270, at=_at[None], device='cuda')
        right_proj = vis_renderer(MESH, R=R, T=T)
        
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
        ax.set_title("Predicted Overlayed")
        # ax.axis('off')

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(left_frame)
        ax.set_title("Predicted Left View Point")
        ax.axis('off')

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(right_frame)
        ax.set_title("Predicted Right View Point")
        ax.axis('off')

        if out_path is not None:
            plt.savefig(os.path.join(out_path, f'{idx:05d}.jpg'), bbox_inches='tight',dpi=100)
        if not skip_interactive:
            plt.show()
        plt.close(fig)

    if out_path is not None:
        ppaths = natsorted(glob.glob(os.path.join(out_path, '*.jpg')))
        _plots = []
        for pp in ppaths:
            _plots.append( imageio.imread(pp) )

        imageio.mimsave(os.path.join(out_path, 'final_result.gif'), _plots, duration=0.1)

def get_mesh(op, color=[0, 1, 0], device='cpu'):

    verts, faces, _ = load_obj(op, device=device)
    tex = torch.ones_like(verts) * torch.tensor(color, device=device)[None]
    tex = tex.unsqueeze(0)
    textures = TexturesVertex(verts_features=tex).to(device)
    
    return Meshes(verts=[verts],faces=[faces.verts_idx],textures=textures)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--results_path', type=str, required=True, help="Path to outputs to visualize")
    parser.add_argument('--out_path', type=str, default=None, help="Path to save visualizations")
    parser.add_argument('--skip_interactive', action="store_true", help="Whether to skip interactive display of images")
    parser.add_argument('--seed', type=int, default=0, help="seed for prngs")
    parser.add_argument('--step', type=int, default=2, help="step to sample frames while loading")

    return parser.parse_args()

def main():
    args = parse_args()

    # seeding
    set_seed(args.seed)

    IMAGE_PATH = os.path.join(args.data_path, 'frames')

    gt_data = read_data(args.data_path)
    focal_x = gt_data['focal']
    focal_y = gt_data['focal']

    images = utils.get_frames(IMAGE_PATH, args.step)
    N, H, W, _ = images.shape

    # load objects
    opaths = natsorted(glob.glob(os.path.join(args.results_path, '*_obj.obj')))
    objs = []
    for op in opaths:
        obj_mesh = get_mesh(op, color=[0, 1, 0], device=device)
        objs.append(obj_mesh)
    
    # load smpls
    spaths = natsorted(glob.glob(os.path.join(args.results_path, '*_smpl.obj')))
    smpls = []
    for sp in spaths:
        smpl_mesh = get_mesh(sp, color=[1, 0, 0],  device=device)
        smpls.append(smpl_mesh)

    assert len(images) == len(objs) == len(smpls)

    # concat
    models = []
    for ob, sm in zip(objs, smpls):
        MESH = join_meshes_as_scene([ob, sm])
        models.append(MESH)

    phong_renderer = initialize_render(device, focal_x=focal_x, focal_y=focal_y, 
                                                        img_square_size=max(H, W))

    visualize_optimal_poses_humans_video(models, images, phong_renderer, args.out_path, args.skip_interactive)

if __name__ == '__main__':
    main()