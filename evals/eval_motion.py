import numpy as np
from numpy import dot
from numpy.linalg import norm

import os
import json
import argparse

# Evalaute the motion type
def evalType(pred_type, gt_type):
    if pred_type == gt_type:
        return True
    else:
        return False

# Evalaute the axis, return the angle difference in degree
def evalAxisDir(pred_axis, gt_axis):
    pred_axis = np.array(pred_axis)
    gt_axis = np.array(gt_axis)

    if np.sum(pred_axis**2) == 0:
        raise ValueError("Pred Axis is not a vector")

    if np.sum(gt_axis**2) == 0:
        raise ValueError("GT Axis is not a vector")

    axis_similarity = dot(gt_axis, pred_axis) / (
        norm(gt_axis) * norm(pred_axis)
    )
    if axis_similarity < 0:
        axis_similarity = -axis_similarity
    axis_similarity = min(axis_similarity, 1.0)
    ## dtAxis used for evaluation metric MD
    axis_similarity = np.arccos(axis_similarity) / np.pi * 180

    return axis_similarity

# Evalaute the axis, return the angle difference in degree
def evalDirection(pred_axis, gt_axis):
    pred_axis = np.array(pred_axis)
    gt_axis = np.array(gt_axis)

    if np.sum(pred_axis**2) == 0:
        raise ValueError("Pred Axis is not a vector")

    if np.sum(gt_axis**2) == 0:
        raise ValueError("GT Axis is not a vector")

    axis_similarity = dot(gt_axis, pred_axis) / (
        norm(gt_axis) * norm(pred_axis)
    )
    axis_similarity = np.clip(axis_similarity, -1., 1.0)
    ## dtAxis used for evaluation metric MD
    axis_similarity = np.arccos(axis_similarity) / np.pi * 180

    return axis_similarity

# Evaluate the origin, return the distance from pred origin to the gt axis line
def evalAxisOrig(pred_axis, gt_axis, pred_origin, gt_origin):
    pred_axis = np.array(pred_axis)
    gt_axis = np.array(gt_axis)
    pred_origin = np.array(pred_origin)
    gt_origin = np.array(gt_origin)

    p = pred_origin - gt_origin
    axis_line_similarity = np.linalg.norm(
        np.cross(p, gt_axis)
    ) / np.linalg.norm(gt_axis)

    return axis_line_similarity

def evalMotionState(pred_state, gt_state, method):

    if method != 'd3d' and method != 'ditto':
        try:
            closed_idx = np.where(gt_state == 0)[0][0]
        except:
            closed_idx = np.argmin(gt_state)
        pred_state = (pred_state - pred_state[closed_idx]) % 360

    err = np.abs((gt_state - pred_state))
    return err

def main(args):
    
    with open(args.videos_file, 'r') as fp:
        videos = [x.strip() for x in fp.readlines()]

    if not os.path.exists(args.out_dir):
        raise Exception("Enter correct directory. Or Run recon eval first")

    axis_error = []
    dir_error = []
    origin_error = []
    motion_state_error = []
    motion_vids = []

    for vid in videos:

        gt_path = os.path.join(args.gt_dir, vid, 'gt_axis_origin.json')
        with open(gt_path, 'r') as fp:
            gt_data = json.load(fp)

        pred_path = os.path.join(args.pred_dir, vid, 'axis_origin_pred.json')
        with open(pred_path, 'r') as fp:
            pred_data = json.load(fp)

        if args.method == '3dadn':
            if type(pred_data['origin']) != list:
                continue 

            motion_vids.append(vid)

        # loading motion state
        # gt
        with open(os.path.join(args.gt_dir, vid, 'jointstate.txt'), 'r') as fp:
            gt_motion_state = [float(x.strip()) for x in fp.readlines() if x.strip() != '']
            gt_motion_state = np.array(gt_motion_state[::args.step]) 

        # pred
        if args.method == 'cubeopt':
            pred_motion_state = np.array(pred_data['motion_state']).astype(np.float32) * 180. / np.pi # converting to degrees
        elif args.method == 'd3d':
            pred_motion_state = np.array(pred_data['motion_state'][::args.step]).astype(np.float32) 
        elif args.method == '3dadn':
            pred_motion_state = np.array(pred_data['motion_state'][::args.step]).astype(np.float32) * 180. / np.pi
        elif args.method == 'ditto':
            pred_motion_state = np.array(pred_data['motion_state'][::args.step]).astype(np.float32) * 180. / np.pi

        ax_err = evalAxisDir(pred_data['axis'], gt_data['axis'])
        ax_dir_err = evalDirection(pred_data['axis'], gt_data['axis'])
        or_err = evalAxisOrig(pred_axis=pred_data['axis'], gt_axis=gt_data['axis'], 
                                    pred_origin=pred_data['origin'], gt_origin=gt_data['origin'])

        
        state_err = evalMotionState(pred_motion_state, gt_motion_state, method=args.method)

        axis_error.append(ax_err)
        dir_error.append(ax_dir_err)
        origin_error.append(or_err)
        motion_state_error.append(state_err.mean())

        frame_axis_errors = np.ones_like(state_err) * ax_err
        frame_dir_errors = np.ones_like(state_err) * ax_dir_err
        frame_orig_errors = np.ones_like(state_err) * or_err
        frame_state_errors = state_err

        os.makedirs(os.path.join(args.out_dir, 'axis_errors'), exist_ok=True)
        np.save(os.path.join(args.out_dir, 'axis_errors', f'{vid.replace("/", "_")}.npy'), frame_axis_errors)

        os.makedirs(os.path.join(args.out_dir, 'dir_errors'), exist_ok=True)
        np.save(os.path.join(args.out_dir, 'dir_errors', f'{vid.replace("/", "_")}.npy'), frame_dir_errors)

        os.makedirs(os.path.join(args.out_dir, 'orig_errors'), exist_ok=True)
        np.save(os.path.join(args.out_dir, 'orig_errors', f'{vid.replace("/", "_")}.npy'), frame_orig_errors)

        os.makedirs(os.path.join(args.out_dir, 'state_errors'), exist_ok=True)
        np.save(os.path.join(args.out_dir, 'state_errors', f'{vid.replace("/", "_")}.npy'), frame_state_errors)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_file', type=str, required=True, help='path to list of videos to evaluation dir')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--step', type=int, default=2)
    args = parser.parse_args()

    main(args)