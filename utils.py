import os, glob
from natsort import natsorted
import imageio

import numpy as np
import math
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_obj, save_obj

# rendering components
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights,
    PerspectiveCameras
)

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

def get_frames(path, step, resize=False):

    fpaths = natsorted(glob.glob(os.path.join(path, '*')))

    frames = []
    for fp in fpaths[::step]:
        img = imageio.imread(fp)
        h, w, _ = img.shape
        if resize:
            frames.append( cv2.resize(img, (h // 2, w // 2)) )
        else:
            frames.append(img)

    return np.array(frames)

def load_nps(paths, resize=False):
    masks = []
    for mp in paths:
        mk = np.load(mp)
        h, w = mk.shape
        if resize:
            masks.append( cv2.resize(mk, (h // 2, w // 2), interpolation=cv2.INTER_NEAREST) ) #TODO: is it the right way to resize mask?
        else:
            masks.append(mk.astype(np.float32))

    return np.array(masks)

def get_masks(path, step, resize=False):

    mpaths = natsorted(glob.glob(os.path.join(path, '*')))
    return load_nps(mpaths[::step], resize=resize)
    
def get_mask_parts(path, step):

    mpaths = natsorted(glob.glob(os.path.join(path, '*')))
    obj, part1, part2 = mpaths[0::3], mpaths[1::3], mpaths[2::3]

    obj_masks = load_nps(obj[::step])
    p1_masks = load_nps(part1[::step])
    p2_masks = load_nps(part2[::step])

    return obj_masks, p1_masks, p2_masks

def get_smpl_meshes(path, step):

    smpl_paths = natsorted(glob.glob(os.path.join(path, '*')))

    smpl_verts = []
    smpl_faces = []
    for smp in smpl_paths[::step]:
        verts, faces, _ = load_obj(smp)
        smpl_verts.append(verts.unsqueeze(0))
        smpl_faces.append(faces.verts_idx.unsqueeze(0))

    return torch.cat(smpl_verts, dim=0), torch.cat(smpl_faces, dim=0)

def get_smpl_points(path, step):

    smpl_paths = natsorted(glob.glob(os.path.join(path, '*')))

    smpl_points = []
    for smp in smpl_paths[::step]:
        points = np.load(smp)
        smpl_points.append(points)
    
    return np.array(smpl_points)

def visualize_optimal_poses(model, image, mask, vis_renderer, score=0, color=None, vis_num=5):
    """
    Visualizes the 8 best-scoring object poses.

    Args:
        model (PoseOptimizer).
        image_crop (H x H x 3).
        mask (M x M x 3).
        score (float): Mask confidence score (optional).
    """

    fig = plt.figure(figsize=((10, 10)))
    ax1 = fig.add_subplot(2 * (vis_num+1), 2, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Image")

    ax2 = fig.add_subplot(2 * (vis_num+1), 2, 2)
    ax2.imshow(mask)
    ax2.axis("off")
    if score > 0:
        ax2.set_title(f"Mask Conf: {score:.2f}")
    else:
        ax2.set_title("Mask")

    verts = model.transform(idx=0)[:1]
    faces = model.faces[:1]
    if color is None:
        text = TexturesVertex(verts_features=model.verts_rgb.view(model.num_frames * model.num_hypos, -1, 3)[:1])
    else:
        assert type(color) in [tuple, torch.Tensor, np.ndarray]
        _rgb = torch.ones_like(verts) * torch.tensor(color).to(verts.device)[None]
        text = TexturesVertex(verts_features=_rgb.to(verts.device))

    print(f"verts: {verts.shape}, faces: {faces.shape}")
    mesh = Meshes(verts, faces, text)
    projection = vis_renderer(mesh)

    for idx in range(len(projection)):
        ax = fig.add_subplot(2 * (vis_num+1), 2, 2 + 2*idx + 1)
        ax.imshow(projection[0, ..., :3].detach().cpu().numpy())
        ax.set_title("Predicted")
        ax.axis('off')

        ax = fig.add_subplot(2 * (vis_num+1), 2, 2 + 2*idx + 2)
        ax.imshow(image)
        ax.imshow(projection[0, ..., :3].detach().cpu().numpy(), alpha=0.2)
        ax.set_title("Predicted")
        ax.axis('off')

    # plt.savefig(f'output.jpg', bbox_inches='tight',dpi=100)
    plt.show()

def visualize_optimal_poses_video(model, images, masks, vis_renderer, right_renderer, score=0, color=None, out_path=''):
    """
    Visualizes the 8 best-scoring object poses.

    Args:
        model (PoseOptimizer).
        image_crop (N x H x H x 3).
        mask (N x M x M x 3).
        score (float): Mask confidence score (optional).
    """

    os.makedirs(out_path, exist_ok=True)

    for idx in range(images.shape[0]):
        fig = plt.figure(figsize=((20, 4)))
        ax1 = fig.add_subplot(1, 5, 1)
        ax1.imshow(images[idx])
        ax1.axis("off")
        ax1.set_title("Image")

        ax2 = fig.add_subplot(1, 5, 2)
        ax2.imshow(masks[idx])
        ax2.axis("off")
        if score > 0:
            ax2.set_title(f"Mask Conf: {score:.2f}")
        else:
            ax2.set_title("Mask")

        verts = model.transform(idx=idx)[:1]
        faces = model.faces[:1]
        if color is None:
            text = TexturesVertex(verts_features=model.verts_rgb.view(model.num_frames * model.num_hypos, -1, 3)[:1])
        else:
            assert type(color) in [tuple, torch.Tensor, np.ndarray]
            _rgb = torch.ones_like(verts) * torch.tensor(color).to(verts.device)[None]
            text = TexturesVertex(verts_features=_rgb.to(verts.device))

        print(f"verts: {verts.shape}, faces: {faces.shape}")
        mesh = Meshes(verts, faces, text)
        projection = vis_renderer(mesh)

        right_proj = right_renderer(mesh)

        diff = (1280 - 720) // 2
        proj_frame = projection[0,...,:3].detach().cpu().numpy()
        proj_frame = proj_frame[:, diff:-diff]

        right_frame = right_proj[0, ..., :3].detach().cpu().numpy()

        ax = fig.add_subplot(1, 5, 3)
        ax.imshow(proj_frame)
        ax.set_title("Predicted")
        ax.axis('off')

        ax = fig.add_subplot(1, 5, 4)
        ax.imshow(right_frame)
        ax.set_title("Predicted Right View Point")
        ax.axis('off')

        ax = fig.add_subplot(1, 5, 5)
        ax.imshow(images[idx])
        ax.imshow(proj_frame, alpha=0.2)
        ax.set_title("Predicted Overlayed")
        ax.axis('off')

        plt.savefig(os.path.join(out_path, f'{idx:05d}.jpg'), bbox_inches='tight',dpi=100)
        save_obj(os.path.join(out_path, f'{idx:05d}.obj'), verts[0], faces[0])
        plt.close()
    
    # generate gif
    ppaths = natsorted(glob.glob(os.path.join(out_path, '*.jpg')))
    _plots = []
    for pp in ppaths:
        _plots.append( imageio.imread(pp) )

    imageio.mimsave(os.path.join(out_path, 'final_result.gif'), _plots, duration=0.1)

def visualize_optimal_poses_humans_video(model, batch, images, masks, vis_renderer, only_human=False, out_path=''):
    """
    Visualizes the 8 best-scoring object poses.

    Args:
        model (PoseOptimizer).
        image_crop (N x H x H x 3).
        mask (N x M x M x 3).
    """

    os.makedirs(out_path, exist_ok=True)
    for idx in range(images.shape[0]):
        
        if only_human:
            obj_mesh, smpl_mesh = model.render(idx)
        else:
            obj_mesh, smpl_mesh, losses = model.render(idx, batch, _cache=True)

        _loss_str = ''
        if not only_human:
            for k, v in losses.items():
                _loss_str += f'{k} : {v:0.4f} \n'

        fig = plt.figure(figsize=((20, 4)))

        ax = fig.add_subplot(1, 7, 1)
        ax.imshow(np.ones(images[idx].shape))
        ax.axis("off")
        ax.text(0., 0.5, _loss_str, fontsize=12, horizontalalignment='left',
                            verticalalignment='center', transform=ax.transAxes)

        ax1 = fig.add_subplot(1, 7, 2)
        ax1.imshow(images[idx])
        ax1.axis("off")
        ax1.set_title("Image")

        ax2 = fig.add_subplot(1, 7, 3)
        ax2.imshow(masks[idx])
        ax2.axis("off")
        ax2.set_title("Mask")

        
        if only_human:
            mesh = smpl_mesh
        else:
            mesh = join_meshes_as_scene([obj_mesh, smpl_mesh])

        # camera is almost at the center (distance can't be zero for diff render)
        R, T = look_at_view_transform(0.1, 0.0, 0.0,device='cuda')
        T[0,2] = 0.0  # manually set to zero
        projection = vis_renderer(mesh, R=R, T=T)
        # left - right viewpoint
        bbox = mesh.get_bounding_boxes()
        _at = bbox[0].mean(dim=1)
        R, T = look_at_view_transform(_at[-1], 0, 90, at=_at[None], device='cuda')
        left_proj = vis_renderer(mesh, R=R, T=T)

        R, T = look_at_view_transform(_at[-1], 0, 270, at=_at[None], device='cuda')
        right_proj = vis_renderer(mesh, R=R, T=T)

        proj_frame = projection[0,...,:3].detach().cpu().numpy()
        H, W, _ = images[idx].shape
        if H > W:
            diff = (H - W) // 2
            proj_frame = proj_frame[:, diff:-diff]
        else:
            diff = (W - H) // 2
            proj_frame = proj_frame[diff:-diff, :]

        left_frame = left_proj[0, ..., :3].detach().cpu().numpy()
        right_frame = right_proj[0, ..., :3].detach().cpu().numpy()

        ax = fig.add_subplot(1, 7, 4)
        ax.imshow(proj_frame)
        ax.set_title("Predicted")
        ax.axis('off')

        ax = fig.add_subplot(1, 7, 5)
        ax.imshow(left_frame)
        ax.set_title("Predicted Left View Point")
        ax.axis('off')

        ax = fig.add_subplot(1, 7, 6)
        ax.imshow(right_frame)
        ax.set_title("Predicted Right View Point")
        ax.axis('off')

        ax = fig.add_subplot(1, 7, 7)

        ax.imshow(images[idx])
        ax.imshow(proj_frame, alpha=0.4 )
        ax.set_title("Predicted Overlayed")
        ax.axis('off')

        print(os.path.join(out_path, f'{idx:05d}.jpg'))
        plt.savefig(os.path.join(out_path, f'{idx:05d}.jpg'), bbox_inches='tight',dpi=100)
        if obj_mesh:
            save_obj(os.path.join(out_path, f'{idx:05d}_obj.obj'), obj_mesh.verts_list()[0], obj_mesh.faces_list()[0])
        if smpl_mesh:
            save_obj(os.path.join(out_path, f'{idx:05d}_smpl.obj'), smpl_mesh.verts_list()[0], smpl_mesh.faces_list()[0])
        plt.close()
    
    # generate gif
    ppaths = natsorted(glob.glob(os.path.join(out_path, '*.jpg')))
    _plots = []
    for pp in ppaths:
        _plots.append( imageio.imread(pp) )

    imageio.mimsave(os.path.join(out_path, 'final_result.gif'), _plots, duration=0.1)

def get_moving_part_cuboid(path):

    objects = glob.glob(os.path.join(path, '*_obj.obj'))
    for ob in objects:

        verts, faces, _ = load_obj(ob)

        _verts = verts[8:]
        _faces = faces.verts_idx[12:] - 8

        save_obj(ob.replace('_obj.obj', '_part.obj'), verts=_verts, faces=_faces)

def generate_point_cloud_from_depth(depth, cx, cy, fx, fy):

    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.
    cx - x input have
    NaN for the z-coordinate in the result.

    """

    device = depth.device
    bs, rows, cols = depth.shape
    # print(bs, rows, cols)
    #depth = depth.T
    c, r = torch.meshgrid(torch.arange(cols), torch.arange(rows), indexing='xy')
    c = c.unsqueeze(0).repeat(bs, 1, 1).float().to(device)
    r = r.unsqueeze(0).repeat(bs, 1, 1).float().to(device)
    valid = (depth > 0.5)
    
    # print(c.shape, r.shape, valid.shape)
    # print(depth.dtype, c.dtype, r.dtype)

    z = torch.where(valid, depth, depth * torch.nan)
    x = torch.where(valid, z * (c - cx) / fx, z * 0.)
    y = torch.where(valid, -z * (r - cy) / fy, z * 0.)

    return torch.stack([x, y, -z], axis=3) 

def visualize_point_cloud(points):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())

    o3d.visualization.draw_geometries([pcd])

def visualize_2point_cloud(points_a, points_b):
    import open3d as o3d

    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(points_a.detach().cpu().numpy())

    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(points_b.detach().cpu().numpy())

    o3d.visualization.draw_geometries([pcd_a, pcd_b])

# borrowed from d3dhoi
def initialize_render(device, focal_x, focal_y, img_square_size, img_small_size):
    """ initialize camera, rasterizer, and shader. """

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
        faces_per_pixel=50,
        bin_size=None,
        max_faces_per_bin=None,
    )

    rasterizer = MeshRasterizer(
            cameras=camera_sfm_small,
            raster_settings=raster_settings
        )
    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRendererWithFragments(
        rasterizer=rasterizer,
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=img_square_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None,
    )

    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    #lights = DirectionalLights(device=device, direction=((0, 0, 1),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_sfm,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=camera_sfm, lights=lights)
    )

    return silhouette_renderer, phong_renderer, rasterizer