# Based on https://github.com/google/lasr/blob/main/scripts/eval_mesh.py
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
sys.path.insert(0,'third_party')
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

import chamfer3D.dist_chamfer_3D
import subprocess
import imageio
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import trimesh
from natsort import natsorted

# pytorch3d 
import pytorch3d.ops
import pytorch3d.loss
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import (
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
)
from pytorch3d.renderer.cameras import OrthographicCameras

# Calculate the angle between two rotation matrix (borrow from ANCSH paper)
def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def rot_diff_rad(rot1, rot2):
    theta = np.clip(( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2, a_min=-1.0, a_max=1.0)
    return np.arccos( theta ) % (2*np.pi)

def np_std_err(x, axis=0):
    x = np.asarray(x)
    return np.std(x, axis=axis) / np.sqrt(x.shape[axis])

parser = argparse.ArgumentParser(description='reconstruction evaluation')
parser.add_argument('--videos_file', default='',
                    help='path to list of videos to evaluation dir')
parser.add_argument('--gt_dir', default='',
                    help='path to gt data')
parser.add_argument('--pred_dir', default='',
                    help='path to eval logs')
parser.add_argument('--eval_part', action='store_true',
                    help='evaluate only moving part reconstruction')
parser.add_argument('--method', default='cubeopt',
                    help='method to evaluate')
parser.add_argument('--step', type=int, default=2)
parser.add_argument('--out_dir', type=str, help="Output directory for intermediate outputs")
args = parser.parse_args()

with open(args.videos_file, 'r') as f:
    videos = [x.strip() for x in f.readlines()]

for vid in videos:

    GTDIR_PART = os.path.join(args.gt_dir, vid, 'PartMeshCorrect')
    GTDIR = os.path.join(args.gt_dir, vid, 'MeshCorrect')

    TESTDIR = os.path.join(args.pred_dir, vid)
    OUTDIR = os.path.join(TESTDIR, f'ReconEvalOut_P{args.eval_part}')
    os.makedirs(OUTDIR, exist_ok=True)

    if args.method=='cubeopt':
        pred_part_meshes = [i for i in sorted( glob.glob('%s/*_part.obj'%(TESTDIR)) )]
        pred_meshes = [i for i in sorted( glob.glob('%s/*_obj.obj'%(TESTDIR)) )]
    elif args.method=='d3d':
        pred_part_meshes = [i for i in sorted( glob.glob('%s/part_meshes/*.obj'%(TESTDIR)) )][::args.step]
        pred_meshes = [i for i in sorted( glob.glob('%s/*.obj'%(TESTDIR)) )][::args.step] 
    elif args.method=='3dadn':
        pred_meshes = [i for i in sorted( glob.glob('%s/Mesh_Preds/*.obj'%(TESTDIR)) )]
        pred_part_meshes = pred_meshes
    elif args.method=='ditto':
        pred_meshes = [i for i in sorted( glob.glob('%s/*_obj.obj'%(TESTDIR)) )]
        pred_part_meshes = [i for i in sorted( glob.glob('%s/*_part.obj'%(TESTDIR)) )]
    elif args.method=='lasr':
        TESTDIR = TESTDIR + '-1'
        pred_meshes = [i for i in sorted( glob.glob('%s/pred*.ply'%(TESTDIR)),key=lambda x: int(x.split('pred')[1].split('.ply')[0]) )][::args.step]
        pred_part_meshes = pred_meshes
    elif args.method == 'viser':
        TESTDIR = TESTDIR + '-1003-1'
        print('%s/%s-mesh*.obj'%(TESTDIR, vid.replace('/', '_')))
        pred_meshes = [i for i in natsorted( glob.glob('%s/%s-vp1pred*.obj'%(TESTDIR, vid.replace('/', '_'))))][::args.step]
        pred_part_meshes = pred_meshes
    else:exit()
    
    if args.method == 'cubeopt':
        gt_meshes =  [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR)) )][::args.step] 
        gt_part_meshes = [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR_PART)) )][::args.step] 
    elif args.method == '3dadn':
        gt_meshes =  [trimesh.load(f'{GTDIR}/{os.path.basename(i)}', process=False) for i in pred_meshes]
        gt_part_meshes = [trimesh.load(f'{GTDIR_PART}/{os.path.basename(i)[:-4]}_part.obj', process=False) for i in pred_meshes]
    elif args.method == 'ditto':
        gt_meshes =  [trimesh.load(f'{GTDIR}/{os.path.basename(i).replace("_obj", "")}', process=False) for i in pred_meshes]
        gt_part_meshes = [trimesh.load(f'{GTDIR_PART}/{os.path.basename(i).replace("_obj", "")[:-4]}_part.obj', process=False) for i in pred_meshes]
    elif args.method == 'd3d':
        gt_meshes =  [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR)) )][::args.step]
        gt_part_meshes = [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR_PART)) )][::args.step]
    elif args.method == 'lasr':
        gt_meshes =  [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR)) )][::args.step]
        gt_part_meshes = [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR_PART)) )][::args.step]
    elif args.method == 'viser':
        gt_meshes =  [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR)) )][::args.step]
        gt_part_meshes = [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(GTDIR_PART)) )][::args.step]
    
    assert(len(gt_meshes) == len(pred_meshes))

    device = torch.device("cuda:0") 
    cameras = OrthographicCameras(device = device)
    lights = PointLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((1.0, 1.0, 1.0),),
    )
    renderer_softtex = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=512,cull_backfaces=True)),
            shader=SoftPhongShader(device = device,cameras=cameras, lights=lights)
    )
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    cds = []
    part_cds = []
    norms=[]
    frames=[]
    pose_rot_errors = []
    pose_trans_errors = []
    pose_scale_errors = []
    for i in range(len(gt_meshes)):

        mesh1 = trimesh.load(pred_meshes[i], process=False)
        part_mesh1 = trimesh.load(pred_part_meshes[i], process=False)

        # load remeshed 
        if args.method=='lasr':
            import subprocess
            mesh1.export('tmp/input.obj')
            print(subprocess.check_output(['Manifold/build/manifold', 'tmp/input.obj', 'tmp/output.obj', '10000']))
            mesh1 = trimesh.load('tmp/output.obj')
        mesh2 = gt_meshes[i]
        part_mesh2 = gt_part_meshes[i]

        trimesh.repair.fix_inversion(mesh1)
        trimesh.repair.fix_inversion(mesh2)

        if mesh1.vertices.shape[0] > 10000:
            mesh1 = trimesh.convex.convex_hull(mesh1)
        if part_mesh1.vertices.shape[0] > 10000:
            part_mesh1 = trimesh.convex.convex_hull(part_mesh1)

        X0 = torch.Tensor(mesh1.vertices[None] ).to(device)
        X1 = torch.Tensor(part_mesh1.vertices[None]).to(device)

        Y0 = torch.Tensor(mesh2.vertices[None] ).to(device)
        Y1 = torch.Tensor(part_mesh2.vertices[None]).to(device)

        if args.method=='lasr':
            cam = np.loadtxt('%s/cam%d.txt'%(TESTDIR,i))
            Rmat =  torch.Tensor(cam[None,:3,:3]).to(device)
            X0 = X0.matmul(Rmat)
            X1 = X1.matmul(Rmat)
        elif args.method=='viser':
            cam = np.loadtxt('%s/%s-cam%d.txt'%(TESTDIR, vid.replace('/', '_'),i))
            Rmat =  torch.Tensor(cam[None,:3,:3]).to(device)
            X0 = X0.matmul(Rmat)
            X1 = X1.matmul(Rmat)
        elif args.method=='smalify':
            X0[:,:,1:] *= -1
            X1[:,:,1:] *= -1

        if args.method == 'lasr':
            X0[:,:,1:] *= -1
            X1[:,:,1:] *= -1   
        elif args.method == 'viser':
            X0[:,:,1:] *= -1
            X1[:,:,1:] *= -1
    
        # normalize to have extent 10 
        y_mean = Y0.mean(1,keepdims=True)
        Y0 = Y0 - y_mean
        max_dis_y = (Y0 - Y0.permute(1,0,2)).norm(2,-1).max()
        Y0 = 10* Y0 / max_dis_y 
        Y0 = Y0 + y_mean

        y_mean1 = Y1.mean(1,keepdims=True)
        Y1 = Y1 - y_mean1
        max_dis_y1 = (Y1 - Y1.permute(1,0,2)).norm(2,-1).max()
        Y1 = 10* Y1 / max_dis_y1 
        Y1 = Y1 + y_mean1
        
        x_mean = X0.mean(1,keepdims=True)
        X0 = X0 - x_mean
        if args.method=='pifuhd' or args.method=='lasr':
            meshtmp = pytorch3d.structures.meshes.Meshes(verts=X0, faces=torch.Tensor(mesh1.faces[None]).to(device))
            Xtmp = pytorch3d.ops.sample_points_from_meshes(meshtmp, 10000) 
            max_dis_x = (Xtmp - Xtmp.permute(1,0,2)).norm(2,-1).max()
        else:
            max_dis_x = (X0 - X0.permute(1,0,2)).norm(2,-1).max()
        X0 = 10* X0 / max_dis_x 
        X0 = X0 + x_mean

        x_mean1 = X1.mean(1,keepdims=True)
        X1 = X1 - x_mean1
        if args.method=='pifuhd' or args.method=='lasr':
            meshtmp = pytorch3d.structures.meshes.Meshes(verts=X1, faces=torch.Tensor(part_mesh1.faces[None]).to(device))
            Xtmp = pytorch3d.ops.sample_points_from_meshes(meshtmp, 10000) 
            max_dis_x1 = (Xtmp - Xtmp.permute(1,0,2)).norm(2,-1).max()
        else:
            max_dis_x1 = (X1 - X1.permute(1,0,2)).norm(2,-1).max()
        X1 = 10* X1 / max_dis_x1 
        X1 = X1 + x_mean1

        meshx = pytorch3d.structures.meshes.Meshes(verts=X0, faces=torch.Tensor(mesh1.faces[None]).cuda())
        meshy = pytorch3d.structures.meshes.Meshes(verts=Y0, faces=torch.Tensor(mesh2.faces[None]).cuda())
        X = pytorch3d.ops.sample_points_from_meshes(meshx, 10000) 
        Y = pytorch3d.ops.sample_points_from_meshes(meshy, 10000) 

        sol1 = pytorch3d.ops.iterative_closest_point(X,Y,estimate_scale=False,max_iterations=10000)

        meshx1 = pytorch3d.structures.meshes.Meshes(verts=X1, faces=torch.Tensor(part_mesh1.faces[None]).cuda())
        meshy1 = pytorch3d.structures.meshes.Meshes(verts=Y1, faces=torch.Tensor(part_mesh2.faces[None]).cuda())
        X_part = pytorch3d.ops.sample_points_from_meshes(meshx1, 10000) 
        Y_part = pytorch3d.ops.sample_points_from_meshes(meshy1, 10000) 

        sol1_part = pytorch3d.ops.iterative_closest_point(X_part,Y_part,estimate_scale=False,max_iterations=10000)
        
        pose_R = sol1.RTs.R.detach().cpu().numpy()[0]
        pose_T = sol1.RTs.T.detach().cpu().numpy()[0]

        rot_err = rot_diff_degree(pose_R, torch.eye(3))
        trans_err = np.linalg.norm(pose_T)
        scale_err = max_dis_y / max_dis_x
        if scale_err > 1:
            scale_err = 1. / scale_err
        scale_err = 1. - scale_err
        scale_err = scale_err.detach().cpu().numpy()
        
        X0 = (sol1.RTs.s*X0).matmul(sol1.RTs.R)+sol1.RTs.T[:,None]
        X1 = (sol1_part.RTs.s*X1).matmul(sol1_part.RTs.R)+sol1_part.RTs.T[:,None]

        # evaluation
        meshx = pytorch3d.structures.meshes.Meshes(verts=X0, faces=torch.Tensor(mesh1.faces[None]).to(device))
        meshy = pytorch3d.structures.meshes.Meshes(verts=Y0, faces=torch.Tensor(mesh2.faces[None]).to(device))
        X, nx= pytorch3d.ops.sample_points_from_meshes(meshx, 10000,return_normals=True)
        Y, ny= pytorch3d.ops.sample_points_from_meshes(meshy, 10000,return_normals=True)
        cd,norm = pytorch3d.loss.chamfer_distance(X,Y, x_normals=nx,y_normals=ny)
        raw2,raw_cd,_,_ = chamLoss(X,Y0)  # this returns distance squared

        # evaluation
        meshx1 = pytorch3d.structures.meshes.Meshes(verts=X1, faces=torch.Tensor(part_mesh1.faces[None]).to(device))
        meshy1 = pytorch3d.structures.meshes.Meshes(verts=Y1, faces=torch.Tensor(part_mesh2.faces[None]).to(device))
        X_part, nx1= pytorch3d.ops.sample_points_from_meshes(meshx1, 10000,return_normals=True)
        Y_part, ny1= pytorch3d.ops.sample_points_from_meshes(meshy1, 10000,return_normals=True)
        cd1,norm1 = pytorch3d.loss.chamfer_distance(X_part,Y_part, x_normals=nx1,y_normals=ny1)

        # error render    
        cm = plt.get_cmap('plasma')
        color_cd = torch.Tensor(cm(2*np.asarray(raw_cd.cpu()[0]))).to(device)[:,:3][None]
        verts = Y0/(1.05*Y0.abs().max()); verts[:,:,0] *= -1; verts[:,:,-1] *= -1; verts[:,:,-1] -= (verts[:,:,-1].min()-1)

        mesh = Meshes(verts=verts, faces=torch.Tensor(mesh2.faces[None]).to(device),textures=TexturesVertex(verts_features=color_cd))
        errimg = renderer_softtex(mesh)[0,:,:,:3]
        
        # shape render
        color_shape = torch.zeros_like(color_cd); color_shape += 0.5
        mesh = Meshes(verts=verts, faces=torch.Tensor(mesh2.faces[None]).to(device),textures=TexturesVertex(verts_features=color_shape))
        imgy = renderer_softtex(mesh)[0,:,:,:3]
        
        # shape render
        color_shape = torch.zeros_like(X0); color_shape += 0.5
        verts = X0/(1.05*Y0.abs().max()); verts[:,:,0] *= -1; verts[:,:,-1] *= -1; verts[:,:,-1] -= (verts[:,:,-1].min()-1)
        mesh = Meshes(verts=verts, faces=torch.Tensor(mesh1.faces[None]).to(device),textures=TexturesVertex(verts_features=color_shape))
        imgx = renderer_softtex(mesh)[0,:,:,:3]

        img = np.clip(255*np.asarray(torch.cat([imgy, imgx,errimg],1).cpu()),0,255).astype(np.uint8)
        cv2.imwrite('%s/cd-%06d.png'%(OUTDIR,i),img[:,:,::-1])
        cv2.imwrite('%s/gt-%06d.png'%(OUTDIR,i),img[:,:512,::-1])
        cv2.imwrite('%s/pd-%06d.png'%(OUTDIR,i),img[:,512:1024,::-1])
        cv2.imwrite('%s/cd-%06d.png'%(OUTDIR,i),img[:,1024:,::-1])

        cds.append(np.asarray(cd.cpu()))
        part_cds.append(np.asarray(cd1.cpu()))
        norms.append(np.asarray(norm.cpu()))
        frames.append(img)
        pose_rot_errors.append(rot_err)
        pose_trans_errors.append(trans_err)
        pose_scale_errors.append(scale_err)

    if len(cds) == 0:
        continue
    print('%s: %.4f +- %.4f'%(vid, np.mean(cds), np_std_err(cds)))
    
    with open('%s/output.txt'%(OUTDIR), 'w') as f:
        f.write('ChamferDist: %.4f, Norms:  %.4f \n'%(np.mean(cds),np_std_err(cds)))
    imageio.mimsave('%s/output.gif'%(OUTDIR), frames, duration=5./len(frames))

    os.makedirs(os.path.join(args.out_dir, 'cds'), exist_ok=True)
    np.save(os.path.join(args.out_dir, 'cds', f"{vid.replace('/', '_')}.npy"), cds)

    os.makedirs(os.path.join(args.out_dir, 'part_cds'), exist_ok=True)
    np.save(os.path.join(args.out_dir, 'part_cds', f"{vid.replace('/', '_')}.npy"), part_cds)

    os.makedirs(os.path.join(args.out_dir, 'rot_errors'), exist_ok=True)
    np.save(os.path.join(args.out_dir, 'rot_errors', f"{vid.replace('/', '_')}.npy"), pose_rot_errors)

    os.makedirs(os.path.join(args.out_dir, 'trans_errors'), exist_ok=True)
    np.save(os.path.join(args.out_dir, 'trans_errors', f"{vid.replace('/', '_')}.npy"), pose_trans_errors)

    os.makedirs(os.path.join(args.out_dir, 'scale_errors'), exist_ok=True)
    np.save(os.path.join(args.out_dir, 'scale_errors', f"{vid.replace('/', '_')}.npy"), pose_scale_errors)