import os, sys
sys.path.append('../')
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import json
import glob

import torch

import open3d as o3d
import json
import numpy as np
from src.utils.joint_estimation import aggregate_dense_prediction_r

from hydra.experimental import initialize, compose
import hydra

from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils.misc import sample_point_cloud

import argparse

def sum_downsample_points(point_list, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    points = np.concatenate([np.asarray(x.points) for x in point_list], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)

def visualize_pairs(pcds):
    colors = [[1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 1, 0], [1, 1, 1]]
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    for i in range(len(pcds)):
        pcds[i].paint_uniform_color(colors[i])

    o3d.visualization.draw_geometries(pcds + [coordinate])

def main(root, dest, frame1, frame2):

    visalize_middle = False
    # read data

    pcd1 = o3d.io.read_point_cloud(f'{root}/{frame1}.pcd')
    pcd2 = o3d.io.read_point_cloud(f'{root}/{frame2}.pcd')

    original_pcd1 = o3d.geometry.PointCloud(pcd1)
    original_pcd2 = o3d.geometry.PointCloud(pcd2)

    if visalize_middle:
        print(len(original_pcd1.points), len(original_pcd2.points))
        visualize_pairs([original_pcd1, original_pcd2])

    pc1 = np.asarray(pcd1.points)
    pc2 = np.asarray(pcd2.points)
    bound_max = np.maximum(pc1.max(0), pc2.max(0))
    bound_min = np.minimum(pc1.min(0), pc2.min(0))
    center = (bound_max + bound_min) / 2
    scale = (bound_max - bound_min).max() * 1.1
 
    # Normalize the two point clouds
    center_transform = np.eye(4)
    center_transform[:3, 3] = -center
    pcd1.transform(center_transform)
    pcd1.scale(1 / scale, np.zeros((3, 1)))

    pcd2.transform(center_transform)
    pcd2.scale(1 / scale, np.zeros((3, 1)))

    src_pcd = sum_downsample_points([pcd1], 0.02, 50, 0.1)
    dst_pcd = sum_downsample_points([pcd2], 0.02, 50, 0.1)
    if visalize_middle:
        print(len(src_pcd.points), len(dst_pcd.points))
        visualize_pairs([src_pcd, dst_pcd])

    with initialize(config_path='configs/'):
        config = compose(
            config_name='config',
            overrides=[
                'experiment=Ditto_s2m.yaml',
            ], return_hydra_config=True)
    config.datamodule.opt.train.data_dir = 'data/'
    config.datamodule.opt.val.data_dir = 'data/'
    config.datamodule.opt.test.data_dir = 'data/'

    model = hydra.utils.instantiate(config.model)
    ckpt = torch.load('data/Ditto_s2m.ckpt')
    device = torch.device(0)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.eval().to(device)

    generator = Generator3D(
        model.model,
        device=device,
        threshold=0.4,
        seg_threshold=0.5,
        input_type='pointcloud',
        refinement_step=0,
        padding=0.1,
        resolution0=32
    )

    pc_start = np.asarray(src_pcd.points)
    pc_end = np.asarray(dst_pcd.points)

    pc_start, _ = sample_point_cloud(pc_start, 8192)
    pc_end, _ = sample_point_cloud(pc_end, 8192)

    sample = {
        'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
        'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to(device).float()
    }

    count = 0
    while True:
        mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
        if mobile_points_all.size(1) != 0:
            break
        if count > 10:
            return
        count += 1
    
    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)

    static_part = mesh_dict[0].as_open3d
    moving_part = mesh_dict[1].as_open3d

    joint_type_prob = joint_type_logits.sigmoid().mean()
    if joint_type_prob.item()< 0.5:
        motion_type = 'rot'
    else:
        motion_type = 'trans'

    if motion_type == 'rot':
        # axis voting
        joint_r_axis = (
            normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
        joint_r_p2l_vec = (
            normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
        )
        joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
        p_seg = mobile_points_all[0].cpu().numpy()
        pivot_point = p_seg + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]

        (
            joint_axis_pred,
            pivot_point_pred,
            config_pred,
        ) = aggregate_dense_prediction_r(
            joint_r_axis, pivot_point, joint_r_t, method="mean"
        )
    else:
        # axis voting
        joint_p_axis = (
            normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_axis_pred = joint_p_axis.mean(0)
        joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
        config_pred = joint_p_t.mean()
        
        pivot_point_pred = mesh_dict[1].bounds.mean(0)

    motion_state = config_pred
    motion_axis = joint_axis_pred

    motion_origin = np.cross(motion_axis, np.cross(pivot_point_pred, motion_axis))

    # Make the object back
    center_transform = np.eye(4)
    center_transform[:3, 3] = center
    static_part.scale(scale, np.zeros((3, 1)))
    static_part.transform(center_transform)
    moving_part.scale(scale, np.zeros((3, 1)))
    moving_part.transform(center_transform)

    motion_origin = motion_origin * scale + center

    motion_params = {
        "axis": motion_axis.tolist(),
        "origin": motion_origin.tolist(),
        "motion_type": motion_type,
        "motion_state": motion_state.item()
    }

    os.makedirs(os.path.join(dest, 'motion_params'), exist_ok=True)
    with open(os.path.join(dest, 'motion_params', f'{frame1}.json'), 'w') as fp:
        json.dump(motion_params, fp)

    full_mesh = static_part + moving_part

    o3d.io.write_triangle_mesh(os.path.join(dest, f'{frame1}_obj.obj'), full_mesh)
    o3d.io.write_triangle_mesh(os.path.join(dest, f'{frame1}_part.obj'), moving_part)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='val_pcds', help='path to data')
    parser.add_argument('--videos_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.videos_file, 'r') as fp:
        videos = [x.strip() for x in fp.readlines()]

    for vid in videos:
        print('DOING : ', vid)
        frames = sorted(glob.glob(os.path.join(args.data_path, vid, '*')))

        with open(os.path.join(args.data_path, vid, 'jointstate.txt'), 'r') as fp:
            states = [x.strip() for x in fp.readlines()]
            states = [float(x) for x in states if x != '']

        states = np.array(states)
        base_frame = np.argmin(states)

        root = os.path.join(args.data_path, vid, 'PointClouds')
        dest = os.path.join(args.out_path, vid)
        os.makedirs(dest, exist_ok=True)
        for fr in frames:
            
            frame1 = os.path.basename(fr).replace('.pcd', '')
            if os.path.exists(os.path.join(dest, f'{frame1}_obj.obj')):
                continue
            
            frame2 = f'{base_frame:05d}'
            
            main(root, dest, frame1, frame2)
