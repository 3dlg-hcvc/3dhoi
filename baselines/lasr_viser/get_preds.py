import  os
from copy import deepcopy
import json
import glob

import numpy as np
from sklearn.decomposition import PCA
import trimesh

from natsort import natsorted
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_folder', type=str, required=True, help='path to predictions dir of lasr/viser')
parser.add_argument('--videos_file', type=str, required=True, help='path to list of videos to evaluate')
parser.add_argument('--method', type=str, required=True, choices=['lasr', 'viser'], help='method for evaluation')

args = parser.parse_args()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def main():

    with open(args.videos_file, 'r') as fp:
        videos = [x.strip() for x in fp.readlines()]

    for vid in videos:

        all_mesh = []
        all_bone = []
        all_bone_vert = []
        all_cam = []
        
        if args.method == 'lasr':
            TESTDIR = os.path.join(args.testdir, f'{vid}-1')
        else:
            TESTDIR = os.path.join(args.testdir, f'{vid}-1003-1')

        DEST = os.path.join(TESTDIR, 'part_meshes')
        os.makedirs(DEST)

        if args.method == 'lasr':
            mesh_paths = natsorted(glob.glob('%s/pred*.ply'%(TESTDIR)))
        else:
            mesh_paths = natsorted(glob.glob('%s/%s-vp1pred*.obj'%(TESTDIR, vid)))

        for name in mesh_paths:

            fr = int(name.split('/')[-1].split('.')[-2])
        
            try:
                try: 
                    mesh = trimesh.load('%s/pred%d.ply'%(TESTDIR, fr),process=False)
                except: 
                    mesh = trimesh.load('%s/pred%d.obj'%(TESTDIR, fr),process=False)
                trimesh.repair.fix_inversion(mesh)

                all_mesh.append(mesh)
                cam = np.loadtxt('%s/cam%d.txt'%(TESTDIR,fr))
                all_cam.append(cam)
                all_bone.append(trimesh.load('%s/gauss%d.ply'%(TESTDIR, fr),process=False))
                skin = np.load('%s/skin.npy'%(TESTDIR))
            except: print('no mesh found')

        # add bones
        num_original_verts = []
        num_original_faces = []

        for i in range(len(all_mesh)):
            all_mesh[i].visual.vertex_colors[:,-1]=192 # alpha
            num_original_verts.append( all_mesh[i].vertices.shape[0])
            num_original_faces.append( all_mesh[i].faces.shape[0]  ) 
            bone_c = np.unique(np.asarray(all_bone[i].visual.vertex_colors)[:, :3], axis=0)
            idxes_1 = np.where(np.all(np.asarray(all_bone[i].visual.vertex_colors)[:, :3] == bone_c[0], axis=1))[0]
            idxes_2 = np.where(np.all(np.asarray(all_bone[i].visual.vertex_colors)[:, :3] == bone_c[1], axis=1))[0]
            
            bone1_v = np.asarray(all_bone[i].vertices)[idxes_1].mean(0, keepdims=True)
            bone2_v = np.asarray(all_bone[i].vertices)[idxes_2].mean(0, keepdims=True)
            all_bone_vert.append(np.concatenate([bone1_v, bone2_v], axis=0))

        ARROW_ORIG = []
        ARROW_DIR = []
 
        for i in range(len(all_mesh)):

            refmesh = all_mesh[i]

            # getting parts
            verts = np.asarray(all_mesh[i].vertices)
            faces = np.asarray(all_mesh[i].faces)

            dist = 1-skin.T
            part_ass = dist.argmin(axis=1)

            # coloring boundary with black
            boundary = []
            for f in refmesh.faces:
                if np.all(part_ass[f] == 0) or np.all(part_ass[f] == 1):
                    continue
                else:
                    boundary.extend(f[part_ass[f] == 0])
            boundary = np.unique(boundary)

            bound_verts = verts[boundary]

            X = bound_verts
            pca = PCA(n_components=3)
            pca.fit(X)

            orig = np.median(bound_verts, axis=0)
            dir = pca.components_[0]

            ARROW_ORIG.append(orig)
            ARROW_DIR.append(dir)

            # get part meshes
            faces_1 = faces[(part_ass[faces].sum(axis=1) <= 1)]
            faces_2 = faces[(part_ass[faces].sum(axis=1) > 1)]

            mesh1 = trimesh.Trimesh(vertices=verts, faces=faces_1)
            mesh2 = trimesh.Trimesh(vertices=verts, faces=faces_2)

            mesh1.remove_unreferenced_vertices()
            mesh2.remove_unreferenced_vertices()

            _ = mesh1.export(os.path.join(DEST, f'{i:05d}_1.obj'))
            _ = mesh2.export(os.path.join(DEST, f'{i:05d}_2.obj'))

        ANGLES = []
        for idx, (_orig, _dir) in enumerate(zip(ARROW_ORIG, ARROW_DIR)):
            
            bone_verts = deepcopy(all_bone_vert[idx])
            refmesh = deepcopy(all_mesh[idx])
            
            bone1, bone2 = bone_verts
            
            vec1 = bone1 - _orig
            vec2 = bone2 - _orig

            norm1 = np.cross(vec1, _dir)
            norm1 = norm1 / np.linalg.norm(norm1)
            norm2 = np.cross(_dir, vec2)
            norm2 = norm2 / np.linalg.norm(norm2)

            ang = angle_between(norm1, norm2)

            ANGLES.append(ang)

        glob_orig = np.median(ARROW_ORIG, axis=0)
        glob_dir = np.median(ARROW_DIR, axis=0)

        motion_data = {"axis": glob_dir.tolist(), "origin": glob_orig.tolist(), "motion_state": ANGLES}
        with open(os.path.join(TESTDIR, 'axis_origin_pred.json'), 'w') as fp:
            json.dump(motion_data, fp)


if __name__ == '__main__':
    main()
