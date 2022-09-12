# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import natsort
import glob
import open3d as o3d
# rendering components
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights,
    PerspectiveCameras
)
import torch.nn as nn
import math

import matplotlib.pyplot as plt


class MeshRendererWithFragments(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.

    In the forward pass this class returns the `fragments` from which intermediate
    values such as the depth map can be easily extracted e.g.

    .. code-block:: python
        images, fragments = renderer(meshes)
        depth = fragments.zbuf
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)

    def forward(self, meshes_world, **kwargs):
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments

def initialize_render(device, focal_x, focal_y, img_square_size, img_small_size):
    """ initialize camera, rasterizer, and shader. """
    # Initialize an OpenGL perspective camera.
    #cameras = FoVPerspectiveCameras(znear=1.0, zfar=9000.0, fov=20, device=device)
    #cameras = FoVPerspectiveCameras(device=device)
    #cam_proj_mat = cameras.get_projection_transform()
    img_square_center = int(img_square_size/2)
    shrink_ratio = int(img_square_size/img_small_size)
    focal_x_small = int(focal_x/shrink_ratio)
    focal_y_small = int(focal_y/shrink_ratio)
    img_small_center = int(img_small_size/2)

    camera_sfm = PerspectiveCameras(
                focal_length=((focal_x, focal_y),),
                principal_point=((img_square_center, img_square_center),),
                image_size = ((img_square_size, img_square_size),),
                in_ndc=False,
                device=device)

    camera_sfm_small = PerspectiveCameras(
                focal_length=((focal_x_small, focal_y_small),),
                principal_point=((img_small_center, img_small_center),),
                image_size = ((img_small_size, img_small_size),),
                in_ndc=False,
                device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=img_small_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm_small,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_square_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    #lights = DirectionalLights(device=device, direction=((0, 0, 1),))
    phong_renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=camera_sfm, lights=lights)
    )

    return silhouette_renderer, phong_renderer




def merge_meshes(obj_path, device):
    """ helper function for loading and merging meshes. """
    verts_list = torch.empty(0,3)
    faces_list = torch.empty(0,3).long()
    num_vtx = [0]
    num_faces = [0]

    # merge meshes, load in ascending order
    meshes = natsort.natsorted(glob.glob(obj_path+'/final/*_rescaled_sapien.obj'))

    for part_mesh in meshes:
        #print('loading %s' %part_mesh)
        mesh = o3d.io.read_triangle_mesh(part_mesh)
        verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long()
        faces = faces + verts_list.shape[0]
        verts_list = torch.cat([verts_list, verts])
        faces_list = torch.cat([faces_list, faces])
        num_vtx.append(verts_list.shape[0])
        num_faces.append(faces_list.shape[0])

    verts_list = verts_list.to(device)
    faces_list = faces_list.to(device)

    return verts_list, faces_list, num_vtx, num_faces



def load_motion(motions, device):
    """ load rotation axis, origin, and limit. """
    rot_origin = []
    rot_axis = []
    rot_type = []
    limit_a = []
    limit_b = []
    contact_list = []

    # load all meta data
    for idx, key in enumerate(motions.keys()):
        jointData = motions[key]

        # if contains movable parts
        if jointData is not None:
            origin = torch.FloatTensor(jointData['axis']['origin']).to(device)
            axis = torch.FloatTensor(jointData['axis']['direction']).to(device)
            mobility_type = jointData['type']
            if 'contact' in jointData:
                contact_list.append(jointData['contact'])

            # convert to radians if necessary
            if mobility_type == 'revolute':
                mobility_a = math.pi*jointData['limit']['a'] / 180.0
                mobility_b = math.pi*jointData['limit']['b'] / 180.0
            else:
                assert mobility_type == 'prismatic'
                mobility_a = jointData['limit']['a']
                mobility_b = jointData['limit']['b']

            rot_origin.append(origin)
            rot_axis.append(axis)
            rot_type.append(mobility_type)
            limit_a.append(mobility_a)
            limit_b.append(mobility_b)

    return rot_origin, rot_axis, rot_type, limit_a, limit_b, contact_list