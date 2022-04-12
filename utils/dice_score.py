import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()

    if input.dim() == 2:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        # At this time the dim == 3
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])

        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target)


def compute_pre(input: Tensor, target: Tensor):
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(input)
    if total.item() == 0:
        return 0
    result = inter.item() / total.item()
    return result


def compute_rec(input: Tensor, target: Tensor):
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(target)
    if total.item() == 0:
        return 0
    result = inter.item() / total.item()
    return result


def compute_pre_rec(input: Tensor, target: Tensor):
    assert input.size() == target.size()

    pre = 0
    rec = 0
    batch_size = input.shape[0]
    for i in range(batch_size):
        pre += compute_pre(input[i, 0, ...], target[i, 0, ...])
        rec += compute_rec(input[i, 0, ...], target[i, 0, ...])

    return pre / batch_size, rec / batch_size


def recall_fucking_loss(input: Tensor, target: Tensor, epsilon=1e-6, weight=10):
    assert input.size() == target.size()

    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(target)

    return (1 - (inter + epsilon) / (sets_sum + epsilon)) * weight
