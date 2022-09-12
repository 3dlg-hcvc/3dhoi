# Evalaution script for the accuracy

import numpy as np
import argparse
import glob
import functools
from time import time

THRESHOLDS = [[10., 0.5, 0.3]]

def get_parser():
    parser = argparse.ArgumentParser(description="Accuracy Evaluation Code")
    parser.add_argument(
        "--result_dir",
        default=f"/path/to/results",
        metavar="DIR",
        help="path for the results to evaluate",
    )
    return parser

def formatResult(x, metric):
    return f'{metric} : {round(np.mean(x), 3) * 100.:.1f}'

if __name__ == "__main__":
    start = time()
    args = get_parser().parse_args()

    for thresh in THRESHOLDS:

        ROT_THRES, TRANS_THRES, SCALE_THRES = thresh
        RES_DIR = args.result_dir

        # Get all the videos name
        videos = glob.glob(f"{RES_DIR}/cds/*")
        videos = [x.split("/")[-1] for x in videos]
        print(f"There are totally {len(videos)} videos")

        cds_acc = []
        part_cds_acc = []
        recons_acc = []
        rot_acc = []
        trans_acc = []
        scale_acc = []
        pose_acc = []
        O_acc = []
        OA_acc = []
        OAD_acc = []
        OADS_acc = []
        RP_acc = []
        RPOA_acc = []
        RPOADS_acc = []

        for i in range(len(videos)):
            vid_name = videos[i]
            # Load the results
            cds = np.load(f"{RES_DIR}/cds/{vid_name}")
            part_cds = np.load(f"{RES_DIR}/part_cds/{vid_name}")
            rot = np.load(f"{RES_DIR}/rot_errors/{vid_name}")
            trans = np.load(f"{RES_DIR}/trans_errors/{vid_name}")
            scale = np.load(f"{RES_DIR}/scale_errors/{vid_name}")
            orig = np.load(f"{RES_DIR}/orig_errors/{vid_name}")
            axis = np.load(f"{RES_DIR}/axis_errors/{vid_name}")
            dir = np.load(f"{RES_DIR}/dir_errors/{vid_name}")
            state = np.load(f"{RES_DIR}/state_errors/{vid_name}")

            assert (
                len(cds)
                == len(part_cds)
                == len(rot)
                == len(trans)
                == len(scale)
                == len(orig)
                == len(axis)
                == len(dir)
                == len(state)
            )

            cds_acc.append(np.mean(cds <= TRANS_THRES))
            part_cds_acc.append(np.mean(part_cds <= TRANS_THRES))
            recons_acc.append(
                np.mean(np.logical_and(cds <= TRANS_THRES, part_cds <= TRANS_THRES))
            )
            rot_acc.append(np.mean(rot <= ROT_THRES))
            trans_acc.append(np.mean(trans <= TRANS_THRES))
            scale_acc.append(np.mean(scale <= SCALE_THRES))
            pose_acc.append(
                np.mean(
                    functools.reduce(
                        np.logical_and, (
                            rot <= ROT_THRES, trans <= TRANS_THRES, scale <= SCALE_THRES
                        )
                    )
                )
            )
            O_acc.append(np.mean(orig <= TRANS_THRES))
            OA_acc.append(
                np.mean(
                    functools.reduce(np.logical_and, (orig <= TRANS_THRES, axis <= ROT_THRES))
                )
            )
            OAD_acc.append(
                np.mean(
                    functools.reduce(
                        np.logical_and, (
                            orig <= TRANS_THRES, axis <= ROT_THRES, dir <= ROT_THRES
                        )
                    )
                )
            )
            OADS_acc.append(
                np.mean(
                    functools.reduce(
                        np.logical_and, (
                            orig <= TRANS_THRES,
                            axis <= ROT_THRES,
                            dir <= ROT_THRES,
                            state <= ROT_THRES,
                        )
                    )
                )
            )
            RP_acc.append(
                np.mean(
                    functools.reduce(
                        np.logical_and, (
                            cds <= TRANS_THRES,
                            part_cds <= TRANS_THRES,
                            rot <= ROT_THRES,
                            trans <= TRANS_THRES,
                            scale <= SCALE_THRES,
                        )
                    )
                )
            )
            RPOA_acc.append(
                np.mean(
                    functools.reduce(
                        np.logical_and, (
                            cds <= TRANS_THRES,
                            part_cds <= TRANS_THRES,
                            rot <= ROT_THRES,
                            trans <= TRANS_THRES,
                            scale <= SCALE_THRES,
                            orig <= TRANS_THRES,
                            axis <= ROT_THRES,
                        )
                    )
                )
            )
            RPOADS_acc.append(
                np.mean(
                    functools.reduce(
                        np.logical_and, (
                            cds <= TRANS_THRES,
                            part_cds <= TRANS_THRES,
                            rot <= ROT_THRES,
                            trans <= TRANS_THRES,
                            scale <= SCALE_THRES,
                            orig <= TRANS_THRES,
                            axis <= ROT_THRES,
                            dir <= ROT_THRES,
                            state <= ROT_THRES,
                        )
                    )
                )
            )
        
        print('Doing : ', thresh)
        print(formatResult(cds_acc, 'CD-Obj')) 
        print(formatResult(part_cds_acc, 'CD-Part')) 
        print(formatResult(recons_acc, 'Reconstruction')) 
        print(formatResult(rot_acc, 'Rotation')) 
        print(formatResult(trans_acc, 'Translation'))
        print(formatResult(scale_acc, 'Scale')) 
        print(formatResult(pose_acc, 'Pose')) 
        print(formatResult(O_acc, 'Origin')) 
        print(formatResult(OA_acc, 'OA')) 
        print(formatResult(OAD_acc, 'OAD')) 
        print(formatResult(OADS_acc, 'OADS')) 
        print(formatResult(RP_acc, 'RP')) 
        print(formatResult(RPOA_acc, 'RPOA')) 
        print(formatResult(RPOADS_acc, 'RPOADS'))

        stop = time()
        print(str(stop - start) + " seconds")
