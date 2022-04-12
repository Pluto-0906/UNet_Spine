import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, compute_pre_rec


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    precision = 0
    recall = 0

    # iterate over the validation set
    for batch in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        image, mask_true = batch["image"], batch["mask"]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true)
            else:
                mask_pred = (
                    F.one_hot(mask_pred.argmax(dim=1), net.n_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...]
                )
                # compute the Precision and the Recall
                pre, rec = compute_pre_rec(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])
                precision += pre
                recall += rec

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return (
        dice_score / num_val_batches,
        precision / num_val_batches,
        recall / num_val_batches,
    )