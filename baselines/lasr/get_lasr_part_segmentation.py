import  os
from copy import deepcopy
import json


import glob

import numpy as np

import cv2
import argparse
import trimesh

import open3d as o3d
import configparser

parser = argparse.ArgumentParser(description='render mesh')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--video_file', type=str, required=True, help='path to list of videos to evaluation dir')
# parser.add_argument('--seqname', default='camel',
#                     help='sequence to test')
# parser.add_argument('--outpath', default='/data/gengshay/output.gif',
#                     help='output path')
# parser.add_argument('--freeze', dest='freeze', action='store_true',
#                     help='freeze object at frist frame')
args = parser.parse_args()

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                        z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat

def get_arrow(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.02,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.01,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow

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

    with open(args.video_file, 'r') as fp:
        videos = [x.strip() for x in fp.readlines()]

    for vid in videos:
        all_anno = []
        all_mesh = []
        all_bone = []
        all_bone_vert = []
        all_cam = []
        all_fr = []
        
        TESTDIR = os.path.join(args.testdir, f'{vid}-1')
        DEST = os.path.join(TESTDIR, 'part_meshes')
        os.makedirs(DEST)

        config = configparser.RawConfigParser()
        print(vid)
        config.read('configs/%s.config'%vid)
        datapath = str(config.get('data', 'datapath'))
        init_frame = int(config.get('data', 'init_frame'))
        end_frame = int(config.get('data', 'end_frame'))
        dframe = int(config.get('data', 'dframe'))
        for name in sorted(glob.glob('%s/*'%datapath))[::dframe]:
            rgb_img = cv2.imread(name)
            sil_img = cv2.imread(name.replace('JPEGImages', 'Annotations').replace('.jpg', '.png'),0)[:,:,None]
            all_anno.append([rgb_img,sil_img,0,0,name])
            seqname = name.split('/')[-2]
            fr = int(name.split('/')[-1].split('.')[-2])
            all_fr.append(fr)
        
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

        # add bones?
        num_original_verts = []
        num_original_faces = []

        for i in range(len(all_mesh)):
            all_mesh[i].visual.vertex_colors[:,-1]=192 # alpha
            num_original_verts.append( all_mesh[i].vertices.shape[0])
            num_original_faces.append( all_mesh[i].faces.shape[0]  ) 
            bone_c = np.unique(np.asarray(all_bone[i].visual.vertex_colors)[:, :3], axis=0)
            idxes_1 = np.where(np.all(np.asarray(all_bone[i].visual.vertex_colors)[:, :3] == bone_c[0], axis=1))[0]
            idxes_2 = np.where(np.all(np.asarray(all_bone[i].visual.vertex_colors)[:, :3] == bone_c[1], axis=1))[0]
            
            # import pdb
            # pdb.set_trace()
            bone1_v = np.asarray(all_bone[i].vertices)[idxes_1].mean(0, keepdims=True)
            bone2_v = np.asarray(all_bone[i].vertices)[idxes_2].mean(0, keepdims=True)
            all_bone_vert.append(np.concatenate([bone1_v, bone2_v], axis=0))

        # store all the results

        size = len(all_anno)

        for i in range(size):

            refimg, refsil, refkp, refvis, refname = all_anno[i]
            # print('%s'%(refname))
            img_size = max(refimg.shape)
            refmesh = all_mesh[i]

            # getting parts
            verts = np.asarray(all_mesh[i].vertices)
            bones = all_bone_vert[i]

            faces = np.asarray(all_mesh[i].faces)
            print(faces.shape)

            # from sklearn.metrics.pairwise import euclidean_distances
            # dist = euclidean_distances(verts, bones)
            dist = 1-skin.T
            part_ass = dist.argmin(axis=1)

            faces_1 = faces[(part_ass[faces].sum(axis=1) <= 1)]
            faces_2 = faces[(part_ass[faces].sum(axis=1) > 1)]

            mesh1 = trimesh.Trimesh(vertices=verts, faces=faces_1)
            mesh2 = trimesh.Trimesh(vertices=verts, faces=faces_2)

            mesh1.remove_unreferenced_vertices()
            mesh2.remove_unreferenced_vertices()

            _ = mesh1.export(os.path.join(DEST, f'{i:05d}_1.obj'))
            _ = mesh2.export(os.path.join(DEST, f'{i:05d}_2.obj'))

if __name__ == '__main__':
    main()
