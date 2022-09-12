import torch

from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss

class IntersectionPenalty():

    def __init__(self) -> None:
        # Create the search tree
        max_collisions = 8
        sigma = 0.5
        point2plane = False

        self.collision_loss_weight = 1 #e-2
        self.search_tree = BVH(max_collisions=max_collisions)
        self.pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                        point2plane=point2plane,
                                                        vectorized=True)

    def __call__(self, verts, faces):
        
        _fidx = faces.view(faces.shape[0], -1, 1)
        _fidx = _fidx.repeat(1, 1, 3)
        triangles = torch.gather(verts, dim=1, index=_fidx)
        triangles = triangles.view(triangles.shape[0], -1, 3, 3)

        with torch.no_grad():
            collision_idxs = self.search_tree(triangles)

        inter_loss = self.collision_loss_weight * self.pen_distance(triangles, collision_idxs)

        inter_loss = inter_loss 

        return inter_loss

def silhouette_loss(pred_masks, gt_masks):
    _diff = (pred_masks - gt_masks) ** 2
    return (torch.sum(_diff, dim=(2, 3)) / torch.sum(gt_masks, dim=(2, 3))).view(-1), _diff

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss
