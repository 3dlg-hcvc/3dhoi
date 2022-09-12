import numpy as np
import argparse
import glob, os
from time import time

def get_parser():
    parser = argparse.ArgumentParser(description="Accuracy Evaluation Code")
    parser.add_argument(
        "--result_dir",
        default=f"/path/to/optimized/results/",
        metavar="DIR",
        help="path for the results to evaluate",
    )
    return parser

def np_std_err(x):
    return np.std(x) / np.sqrt(len(x))
def formatResult(x, metric):
    return f'{metric} : {round(np.mean(x), 2):.2f} +- {round(np_std_err(x), 2):.2f}'

if __name__ == "__main__":
    start = time()
    args = get_parser().parse_args()

    RES_DIR = os.path.join(args.result_dir)
    # Get all the videos name
    models = glob.glob(f"{RES_DIR}/axis_errors/*")
    models = [x.split("/")[-1] for x in models]
    
    print(f"There are totally {len(models)} videos")

    cds_err = []
    part_cds_err = []
    rot_err = []
    trans_err = []
    scale_err = []
    orig_err = []
    axis_err = []
    dir_err = []
    state_err = []

    for i in range(len(models)):
        model_name = models[i]
        # Load the results
        cds = np.load(f"{RES_DIR}/cds/{model_name}")
        part_cds = np.load(f"{RES_DIR}/part_cds/{model_name}")
        rot = np.load(f"{RES_DIR}/rot_errors/{model_name}")
        trans = np.load(f"{RES_DIR}/trans_errors/{model_name}")
        scale = np.load(f"{RES_DIR}/scale_errors/{model_name}")
        orig = np.load(f"{RES_DIR}/orig_errors/{model_name}")
        axis = np.load(f"{RES_DIR}/axis_errors/{model_name}")
        dir = np.load(f"{RES_DIR}/dir_errors/{model_name}")
        state = np.load(f"{RES_DIR}/state_errors/{model_name}")

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

        cds_err.append(cds.mean())
        part_cds_err.append(part_cds.mean())
        rot_err.append(rot.mean())
        trans_err.append(trans.mean())
        scale_err.append(scale.mean())
        orig_err.append(orig.mean())
        axis_err.append(axis.mean())
        dir_err.append(dir.mean())
        state_err.append(state[np.logical_not(np.isnan(state))].mean())
    
    print(formatResult(cds_err, 'CD-Obj'))
    print(formatResult(part_cds_err, 'CD-Part'))
    print(formatResult(rot_err, 'Rotation'))
    print(formatResult(trans_err, 'Translation'))
    print(formatResult(scale_err, 'Scale')) 
    print(formatResult(orig_err, 'Origin')) 
    print(formatResult(axis_err, 'Axis')) 
    print(formatResult(dir_err, 'Direction'))
    print(formatResult(state_err, 'State'))

    stop = time()
    print(str(stop - start) + " seconds")
