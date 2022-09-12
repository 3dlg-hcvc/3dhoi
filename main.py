import random
from tqdm import tqdm
import argparse
import torch

import os, re, json
import numpy as np

from model import PoseOptimizer
import human_model
import utils

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def set_seed(seed=2021):
    if seed < 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_params(args):
    
    with open(os.path.join(args.out_path, 'params.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="path to dataset")
    parser.add_argument('--out_path', type=str, required=True, help="output path for mesh projections")
    parser.add_argument('--lr', type=float, default=0.05, help="learning rate for the optimization")
    parser.add_argument('--iterations', type=int, default=500, help="number of iterations to run optimization")
    parser.add_argument('--seed', type=int, default=-1, help="seed for prngs")
    parser.add_argument('--step', type=int, default=2, help="step to sample frames while loading")
    parser.add_argument('--no-hoi', dest='hoi', action='store_false', help="set the flag to run without hoi losses")
    parser.add_argument('--no-depth', dest='depth', action='store_false', help="set the flag to run without depth loss")
    parser.add_argument('--no-contact', dest='contact', action='store_false', help="set the flag to run without contact loss")
    parser.add_argument('--desc', type=str, required=True, help="description of the experiment")

    parser.set_defaults(hoi=True)
    parser.set_defaults(depth=True)
    parser.set_defaults(contact=True)

    return parser.parse_args()

def find_optimal_pose(batch, model=None, lr=0.01, iterations=500, run='hum', hoi=True, depth=True, contact=True):

    if model is None:
        raise Exception('Model is not initialized.')

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.75 * iterations), gamma=0.1)

    loop = tqdm(range(iterations))
    for i in loop:

        batch['iter'] = i
        losses = model(batch)
        total_loss = 0.

        if run == 'hum': # optimizing only human
            total_loss += losses['l_points']

        else: # optimizing human and object

            total_loss += float((i > (0.1 * iterations))) * (losses['sil_loss'].mean() + losses['sil_dice'].mean())

            # static part losses
            total_loss += losses['p1_loss'].mean()
            total_loss += losses['p1_dice'].mean()

            # moving part losses
            total_loss += losses['p2_loss'].mean()
            total_loss += losses['p2_dice'].mean()

            # overlap / interpenetration and scale losses
            total_loss += losses['inter_pen'].mean()
            total_loss += losses['sc_loss'].mean()

            # hoi losses
            if hoi:
                total_loss += losses['l_points'].mean()
                if depth:
                    total_loss += losses['avg_depth_loss'].mean()
                if contact:
                    total_loss += losses['contact_curve_loss'].mean()
        
        optimizer.zero_grad()
        loss = total_loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if run == 'hum':
            loop.set_description('Optimizing (%.4f): l_points %.4f' % (total_loss, losses['l_points']))
        else:
            loop.set_description(('Optimizing (loss %.4f): sil_loss %.4f, inter_pen %.4f, contact %.4f'
                                    'p1_loss %.4f, p2_loss %.4f, l_points %.4f, avg_dl %.4f') \
                % (total_loss.mean(), losses['sil_loss'].mean(), losses['inter_pen'].mean(), losses['contact_curve_loss'].mean(),
                    losses['p1_loss'].mean(), losses['p2_loss'].mean(), losses['l_points'].mean(),
                    losses['avg_depth_loss'].mean()))

    if run == 'hum':
        losses = model(batch)
        ret_loss = losses['l_points'].mean()
    else:
        losses = model(batch)
        ret_loss = 0
        ret_loss += losses['sil_loss'].mean()
        ret_loss += losses['sil_dice'].mean()
        ret_loss += losses['p1_loss'].mean()
        ret_loss += losses['p1_dice'].mean()

        # moving part losses
        ret_loss += losses['p2_loss'].mean()
        ret_loss += losses['p2_dice'].mean()
        # hoi losses
        ret_loss += losses['inter_pen'].mean()

        if hoi:
            if depth:
                ret_loss += losses['avg_depth_loss'].mean()
            if contact:
                ret_loss += losses['contact_curve_loss'].mean()

    return model, ret_loss

def correct_image_size(img_h, img_w, low, high):
    # automatically finds a good ratio in the given range
    img_square = max(img_h,img_w)
    img_small = -1
    for i in range(low, high):
        if img_square % i == 0:
            img_small = i
            break
    return img_square, img_small

def main():

    args = parse_args()

    os.makedirs(args.out_path)
    save_params(args)
    # seeding
    set_seed(args.seed)

    IMAGE_PATH = os.path.join(args.data_path, 'frames')
    MASK_PATH = os.path.join(args.data_path, 'object_masks')
    SMPL_MESH_PATH = os.path.join(args.data_path, 'smplmesh')
    SMPL_POINT_PATH = os.path.join(args.data_path, 'smplv2d')
    INFO_PATH = os.path.join(args.data_path, '3d_info.txt')

    # load gt part motion
    gt_partmotion = []
    fp = open(os.path.join(args.data_path, 'jointstate.txt'))
    for i, line in enumerate(fp):
        line = line.strip('\n')
        if utils.isfloat(line) or utils.isint(line):
            gt_partmotion.append(float(line))
    gt_partmotion = np.asarray(gt_partmotion)

    # Infer HOI snippet from GT part motions
    diff = gt_partmotion[:-1] - gt_partmotion[1:]  # (i - i+1)
    if 'storage_prismatic' in args.data_path:
        large_diff = np.where(abs(diff)>0.5)[0]
    else:
        large_diff = np.where(abs(diff)>2)[0]
    care_idx = np.union1d(large_diff, large_diff+1)
    care_idx = np.clip(care_idx, 0, len(gt_partmotion)-1) // args.step
    care_idx = np.unique(care_idx)

    with open(INFO_PATH) as myfile:
        gt_data = [next(myfile).strip('\n') for x in range(14)]
    
    # initialize focal len 
    focal_len = float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[12])[0]) 
    assert(focal_len ==  float(re.findall(r"[-+]?\d*\.\d+|\d+", gt_data[13])[0])) # in pixel (for 1280x720 only)


    images = utils.get_frames(IMAGE_PATH, args.step)
    obj_masks, p1_masks, p2_masks = utils.get_mask_parts(MASK_PATH, args.step)

    N, H, W, _ = images.shape
    images = images[:obj_masks.shape[0]]

    # loading smpls
    smpl_verts, smpl_faces = utils.get_smpl_meshes(SMPL_MESH_PATH, args.step)
    smpl_points = utils.get_smpl_points(SMPL_POINT_PATH, args.step)

    img_square_size, img_small_size = correct_image_size(H, W, low=200, high=300)
    silhouette_renderer, phong_renderer, rasterizer = utils.initialize_render(device, focal_x=focal_len, focal_y=focal_len, 
                                                                                img_square_size=img_square_size, img_small_size=img_small_size)

    hmodel = human_model.PoseOptimizer(num_frames=obj_masks.shape[0], renderer=silhouette_renderer, rasterizer=rasterizer, 
                                smpl_verts=smpl_verts, smpl_faces=smpl_faces, scale=1., focal_length=focal_len, 
                                img_w=W, img_h=H).to(device)

    batch = {
        'iter': 0,
        'points': torch.from_numpy(smpl_points).to(device),
    }
    hmodel, _ = find_optimal_pose(batch, model=hmodel, iterations=args.iterations, run='hum')

    # hand vertices in smpl
    handcontact = [2005, 5466] 

    MOV_IDXES = [0, 0, 0, 0, 2, 2, 4, 4]
    IDXES = [0, 4, 0, 2, 2, 6, 4, 6]
    AXES = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    SCALES = [[1., 0.5], [1., 0.5], [1., 0.5], [1., 0.5], [0.5], [0.5], [0.5], [0.5]]

    all_losses = []
    for num_idx, (align_idx, mov_idx, align_axis, align_scales) in enumerate(zip(IDXES, MOV_IDXES, AXES, SCALES)):
        hand_losses = []
        for hand in handcontact:
            scale_losses = []
            for scale in align_scales:
                batch = {
                    'iter': 0,
                    'rgb': images,
                    'mask_ref': torch.from_numpy(obj_masks).to(device),
                    'points': torch.from_numpy(smpl_points).to(device),
                    'care_idx': care_idx,
                    'handcontact_v': hand,
                }
                if p1_masks is not None:
                    batch['p1_masks_ref'] = torch.from_numpy(p1_masks).to(device)
                if p2_masks is not None:
                    batch['p2_masks_ref'] = torch.from_numpy(p2_masks).to(device)

                set_seed(args.seed)
                model = PoseOptimizer(num_frames=obj_masks.shape[0], num_hypos=1, renderer=silhouette_renderer, rasterizer=rasterizer, 
                                            smpl_verts=smpl_verts, smpl_faces=smpl_faces, scale=1., focal_length=focal_len, 
                                            img_w=W, img_h=H, num_parts=2, smpl_offset=hmodel.smpl_offset.detach(), align_idx=align_idx,
                                            mov_idx=mov_idx, align_axis=[align_axis], mov_scale=scale).to(device)

                model, loss = find_optimal_pose(batch, model=model, lr=args.lr, iterations=args.iterations, run=f'hoi_{num_idx}_{hand}_{scale}',
                                                hoi=args.hoi, depth=args.depth, contact=args.contact)

                # # saving model
                os.makedirs(os.path.join(args.out_path, 'model'), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.out_path, 'model', f'pose_model_{num_idx}_{hand}_{scale}.pt'))

                scale_losses.append(loss.item())

                with torch.no_grad():
                    model._transform_and_cache()
                    utils.visualize_optimal_poses_humans_video(model, batch, images, obj_masks, vis_renderer=phong_renderer, out_path=os.path.join(args.out_path, f'infer_{num_idx}_{hand}_{scale}'))

                del model

            hand_losses.append((np.min(scale_losses), align_scales[np.argmin(scale_losses)]))
        
        min_hand = np.argmin(hand_losses, axis=0)[0]
        all_losses.append([hand_losses[min_hand][0], hand_losses[min_hand][1], handcontact[min_hand]])


    torch.save(hmodel.state_dict(), os.path.join(args.out_path, 'model', 'human_model.pt'))
    np.save(os.path.join(args.out_path, 'model', 'losses_idx.npy'), all_losses)

    # load best model and run inference

    best_idx = np.argmin(all_losses, axis=0)[0]
    best_scale = all_losses[best_idx][1]
    best_hand = all_losses[best_idx][2]
    model = PoseOptimizer(num_frames=obj_masks.shape[0], num_hypos=1, renderer=silhouette_renderer, rasterizer=rasterizer, 
                                    smpl_verts=smpl_verts, smpl_faces=smpl_faces, scale=1., focal_length=focal_len, 
                                    img_w=W, img_h=H, num_parts=2, smpl_offset=hmodel.smpl_offset.detach(), align_idx=best_idx).to(device)
    state_dict = torch.load(os.path.join(args.out_path, 'model', f'pose_model_{best_idx}_{best_hand}_{best_scale}.pt'))
    model.load_state_dict(state_dict)

    batch['handcontact_v'] = best_hand
    with torch.no_grad():
        model._transform_and_cache()
        utils.visualize_optimal_poses_humans_video(model, batch, images, obj_masks, vis_renderer=phong_renderer, out_path=args.out_path)
        utils.get_moving_part_cuboid(args.out_path)

    torch.save(model.state_dict(), os.path.join(args.out_path, 'model', 'pose_model_best.pt'))

    model._transform_and_cache()
    # getting origin and axis
    from pytorch3d.transforms import rotation_6d_to_matrix
    static_R = rotation_6d_to_matrix(model.static_angles).unsqueeze(0).repeat(model.num_frames, 1, 1, 1).view(-1, 3, 3)
    _cubeaxis = model.align_axis.repeat(model.num_frames, 1, 1).transpose(1, 2).to(model.verts.device)
    
    axis = torch.bmm(static_R,  _cubeaxis)
    origin = model.t_verts_cache.view(model.num_frames, 2, 8, 3)[:, 0, model.align_idx:model.align_idx+1]

    axis = axis[0].detach().cpu().numpy().reshape(-1)
    origin = origin[0].detach().cpu().numpy().reshape(-1)

    axis[1:] *= -1
    origin[1:] *= -1

    motion_state = model.rot_angles.detach().cpu().numpy()

    pred_data = {'axis': axis.tolist(), 'origin':origin.tolist(), 'motion_state':motion_state.tolist()}
    with open(os.path.join(args.out_path, 'axis_origin_pred.json'), 'w') as fp:
        json.dump(pred_data, fp)

if __name__ == '__main__':
    main()
