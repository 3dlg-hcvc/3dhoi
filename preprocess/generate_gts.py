# Copyright (c) Facebook, Inc. and its affiliates.
import cv2
import numpy as np
from utils import (
    initialize_render, merge_meshes,
    load_motion
)
import torch
from PIL import Image
from model import JOHMRLite
import os
import glob
import json
from pathlib import Path
import argparse
import re 
import matplotlib.pyplot as plt 

from pytorch3d.io import save_obj
import open3d as o3d

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


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


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

def correct_image_size(img_h, img_w, low, high):
    # automatically finds a good ratio in the given range
    img_square = max(img_h,img_w)
    img_small = -1
    for i in range(low, high):
        if img_square % i == 0:
            img_small = i
            break
    return img_square, img_small

def create_model(gt_data, data_folder, cad_folder):
    """
    create initial models
    """

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

    device = torch.device("cuda:0")
    obj_path = os.path.join(cad_folder, gt_data['cad'])
    verts, faces, vertexSegs, faceSegs = merge_meshes(obj_path, device)
    verts[:,1:] *= -1  # pytorch3d -> world coordinate
    obj_verts = verts.to(device)
    obj_faces = faces.to(device)

    # load motion json file
    with open(os.path.join(cad_folder, gt_data['cad'], 'motion.json')) as json_file:
        motions = json.load(json_file)
    assert len(motions) + 2 == len(vertexSegs)
    rot_o, rot_axis, rot_type, limit_a, limit_b, contact_list = load_motion(motions, device)

    frames = find_files(os.path.join(data_folder, 'frames'), '.jpg')
    image_bg = np.array(Image.open(frames[0]))/255.0
    img_h = image_bg.shape[0]
    img_w = image_bg.shape[1]
    img_square, img_small = correct_image_size(img_h, img_w, 200, 300)

    # render
    silhouette_renderer, _ = initialize_render(device, focal_x, focal_y, img_square, img_small)

    # Model >_<
    model =  JOHMRLite(x_offset, y_offset, z_offset, yaw, pitch, roll, part_motion, obj_size, \
                       obj_verts, obj_faces, silhouette_renderer, gt_data['part'], rot_o, rot_axis, \
                       vertexSegs, faceSegs, rot_type)

    return model, len(frames), img_h, img_w


def display_img(model, index, data_folder):

    frames = find_files(os.path.join(data_folder, 'frames'), '.jpg')
    image_bg = np.array(Image.open(frames[index]))/255.0
    img_h = image_bg.shape[0]
    img_w = image_bg.shape[1]
    _, img_small = correct_image_size(img_h, img_w, 200, 300)

    with torch.no_grad():
        image, depth, part_image, verts, faces, part_verts, part_faces, origin, axis = model(index)
    rgb_mask = image_bg.astype(np.float32) #cv2.addWeighted(objmask.astype(np.float32), 0.5, image_bg.astype(np.float32), 0.5, 0.0)
    
    # TODO: remove
    frame_img = np.zeros((img_small, img_small,3))
    ratio = img_small / max(img_h, img_w)
    img_h = int(img_h * ratio)
    img_w = int(img_w * ratio)
    rgb_mask = cv2.resize(rgb_mask, (img_w, img_h))
    start = int((max(img_h, img_w) - min(img_h, img_w))/2) - 1
    end = start + min(img_h, img_w)
    if img_h > img_w:
        frame_img[:, start:end,  :] = rgb_mask
    else:
        frame_img[start:end, :, :] = rgb_mask
    rgb_mask = frame_img

    return rgb_mask, image, depth[0], part_image, verts, faces, part_verts, part_faces, origin, axis


def point_cloud(depth, cx, cy, fx, fy):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.
    cx - x input have
    NaN for the z-coordinate in the result.

    """
    rows, cols = depth.shape
    #depth = depth.T
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 1.)
    depth = depth #* (np.sqrt(3))
    z = np.where(valid, depth, 0)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, -z * (r - cy) / fy, 0)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    valid = valid.reshape(-1, 1)

    x, y, z = x[valid], y[valid], z[valid]

    return np.stack([x, y, -z], axis=1) #points



parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, help="annotation data folder")
parser.add_argument("--cad_folder", type=str, help="cad data folder")
parser.add_argument("--videos_file", type=str, help="txt file for videos")
args = parser.parse_args()

with open(args.videos_file, 'r') as fp:
    videos = [x.strip() for x in fp.readlines()]

videos_path = []
for vid in videos:
    print(vid)
    cat = vid.split('/')[0]

    videos_path.append((cat, os.path.join(args.data_folder, vid)))

for cat, vid in sorted(videos_path):

    print('Doing: ', vid)
    os.makedirs(os.path.join(vid, 'MeshCorrect'), exist_ok=True)
    os.makedirs(os.path.join(vid, 'PartMeshCorrect'), exist_ok=True)
    os.makedirs(os.path.join(vid, 'PointClouds'), exist_ok=True)
    os.makedirs(os.path.join(vid, 'object_masks'), exist_ok=True)
    
    # get first mask
    mpath = os.path.join(vid, 'gt_mask', '0001_object_mask.npy')
    fmask = np.load(mpath)

    gt_data = read_data(vid)
    model, num_frames, img_h, img_w = create_model(gt_data, vid, cad_folder=os.path.join(args.cad_folder, cat))

    axes = []
    origins = []
    import json
    for index in range(num_frames):
        frame, img_blend, depth, part_image, verts, faces, part_verts, part_faces, orig, ax = display_img(model, index, vid)

        save_obj(os.path.join(vid, 'MeshCorrect', f'{index:05d}.obj'), verts=verts, faces=faces)
        save_obj(os.path.join(vid, 'PartMeshCorrect', f'{index:05d}_part.obj'), verts=part_verts, faces=part_faces)


        part_mask = (part_image > 0.01)
        assert part_mask.shape[0] == fmask.shape[0] and part_mask.shape[1] == fmask.shape[1]

        obj_mask = np.logical_or(fmask, part_mask)

        np.save(os.path.join(vid, 'object_masks', f'{index:05d}_0.npy'), obj_mask) # full obj
        np.save(os.path.join(vid, 'object_masks', f'{index:05d}_1.npy'), fmask) # base
        np.save(os.path.join(vid, 'object_masks', f'{index:05d}_2.npy'), part_mask) # moving

        depth = np.where(depth == -1, 0, depth)

        cx, cy = img_w // 2, img_h // 2
        depth_pc = point_cloud(depth, cx, cy, fx=gt_data['focal'], fy=gt_data['focal']) 
        pc = np.array(depth_pc)
        pc = pc.reshape(-1, 3)
        pc = pc[~np.isnan(pc[:, -1])]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        o3d.io.write_point_cloud(os.path.join(vid, 'PointClouds', f'{index:05d}.pcd'), pcd)
        
        _ax = ax[0].detach().cpu().numpy()
        _orig = orig[0].detach().cpu().numpy()
        _ax[1:] *= -1
        _orig[1:] *= -1
        axes.append(_ax.tolist())
        origins.append(_orig.tolist())

    ax_orig = {'axis' : axes[0], 'origin' : origins[0]}
    with open(os.path.join(vid, 'gt_axis_origin.json'), 'w') as f:
        json.dump(ax_orig, f)