import cv2
import numpy as np
from utils import (
    merge_meshes,
    load_motion
)
import torch
from PIL import Image
from lite_model import JOHMRLite
import os
import glob
import json
from pathlib import Path
import argparse
import re 

from pytorch3d.io import save_obj
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles

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

def read_data_numpy(path):

    params = np.load(os.path.join(path, 'params.npy'), allow_pickle=True).item()

    obj_offset = params['obj_offset']
    [x_off, y_off, z_off] = obj_offset
    [x_dim, y_dim, z_dim] = params['obj_dim']
    obj_rot_angle = params['obj_rot_angle']
    rot_mat = rotation_6d_to_matrix(torch.from_numpy(obj_rot_angle))
    angles = matrix_to_euler_angles(rot_mat, "XYZ").detach().cpu().numpy()
    print(angles.shape)
    [pitch, yaw, roll] = angles[0]

    if 'exp_sfp' in path:
        part_motion = params['part_motion']
    else:
        part_motion = params['part_motion'] * 180 / np.pi

    data = {'part_motion': part_motion,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'x_offset': x_off,
            'y_offset': y_off,
            'z_offset': z_off,
            'obj_size': [x_dim, y_dim, z_dim]}
    return data

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

    device = torch.device("cuda:0")
    obj_path = os.path.join(cad_folder, gt_data['cad'])
    verts, faces, vertexSegs, faceSegs = merge_meshes(obj_path)
    verts[:,1:] *= -1  # pytorch3d -> world coordinate
    obj_verts = verts.to(device)
    obj_faces = faces.to(device)

    # load motion json file
    with open(os.path.join(cad_folder, gt_data['cad'], 'motion.json')) as json_file:
        motions = json.load(json_file)
    assert len(motions) + 2 == len(vertexSegs)
    rot_o, rot_axis, rot_type, _, _, _ = load_motion(motions, device)

    frames = find_files(os.path.join(data_folder, 'frames'), '.jpg')

    # Model >_<
    model =  JOHMRLite(x_offset, y_offset, z_offset, yaw, pitch, roll, part_motion, obj_size, \
                       obj_verts, obj_faces, gt_data['part'], rot_o, rot_axis, \
                       vertexSegs, faceSegs, rot_type)

    return model, len(frames)


def get_preds(model, index):

    with torch.no_grad():
        verts, faces, part_verts, part_faces, rot_o, rot_a = model(index)
    
    return verts, faces, part_verts, part_faces, rot_o, rot_a

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="annotation data folder")
    parser.add_argument("--pred_folder", type=str, help="Predictions folder")
    parser.add_argument("--cad_folder", type=str, help="cad data folder")
    parser.add_argument("--out_folder", type=str, help="output folder")
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.pred_folder, '*')))

    CAT = os.path.basename(args.cad_folder)

    for vid in videos:

        os.makedirs(os.path.join(args.out_folder, f'{CAT}_{os.path.basename(vid)}'), exist_ok=True)

        cad_models = sorted(glob.glob(os.path.join(vid, '*')))
        for cad in cad_models:
            settings = sorted(glob.glob(os.path.join(cad, '*')))

            for _set in settings:
                print('doing : ', _set)
                gt_data = read_data(os.path.join(args.data_folder, os.path.basename(vid)))
                
                pred_data = read_data_numpy(_set)

                pred_data['cad'] = os.path.basename(cad)
                pred_data['focal'] = gt_data['focal']
                pred_data['part'] = gt_data['part']

                model, num_frames = create_model(pred_data, os.path.join(args.data_folder, os.path.basename(vid)), args.cad_folder)

                for index in range(num_frames):
                    verts, faces, part_verts, part_faces, rot_o, rot_a = get_preds(model, index)
                    save_obj(os.path.join(args.out_folder, f'{CAT}_{os.path.basename(vid)}', f'{index:05d}_obj.obj'), verts=verts, faces=faces)
                    save_obj(os.path.join(args.out_folder, f'{CAT}_{os.path.basename(vid)}', f'{index:05d}_part.obj'), verts=part_verts, faces=part_faces)

                pred_data = {'axis': rot_a[0].detach().cpu().numpy().tolist(), 'origin':rot_o[0].detach().cpu().numpy().tolist()}
                with open(os.path.join(args.out_folder, f'{CAT}_{os.path.basename(vid)}', f'axis_origin_pred.json'), 'w') as fp:
                    json.dump(pred_data, fp)
