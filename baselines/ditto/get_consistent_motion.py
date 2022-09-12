import os, glob
import json
import numpy as  np

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--videos_file', type=str, required=True)
    parser.add_argument('--res_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.video_file, 'r') as fp:
        videos = [x.strip() for x in fp.readlines()]

    for vid in videos:

        nframes = len(glob.glob(os.path.join(args.data_path, vid, 'frames', '*')))

        axes = []
        origin = []
        motion_type = []
        motion_state = []

        for fn in range(nframes):
            
            if os.path.exists(os.path.join(args.path, vid, 'motion_params', f'{fn:05d}.json')):
                with open(os.path.join(args.path, vid, 'motion_params', f'{fn:05d}.json'), 'r') as fp:
                    _params = json.load(fp)

                axes.append(_params['axis'])
                origin.append(_params['origin'])
                motion_type.append(_params['motion_type'])
                motion_state.append(_params['motion_state'])
            else:
                motion_state.append(np.nan)

        axes = np.asarray(axes)
        origin = np.asarray(origin)
        
        med_axis = np.median(axes, axis=0)
        med_origin = np.median(origin, axis=0)

        motion_params = {
            "axis": med_axis.tolist(),
            "origin": med_origin.tolist(),
            "motion_type": motion_type,
            "motion_state": motion_state
        }

        with open(os.path.join(args.path, vid, 'axis_origin_pred.json'), 'w') as fp:
            json.dump(motion_params, fp)

if __name__ == '__main__':
    main()
