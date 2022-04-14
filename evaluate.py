import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import compute_pre_rec, dice_coeff, multiclass_dice_coeff


def evaluate(net, dataloader, device, multi_class=False):
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
        image, true_masks = batch["image"], batch["mask"]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device)
        true_masks = (
            F.one_hot(true_masks).permute(0, 3, 1, 2)
            if multi_class
            else true_masks.unsqueeze(dim=1)
        )

        with torch.no_grad():
            # predict the mask
            masks_pred = net(image)
            masks_pred = masks_pred = (
                F.softmax(masks_pred, dim=1)
                if multi_class
                else torch.sigmoid(masks_pred)
            )

            # compute dice score
            dice_score += multiclass_dice_coeff(masks_pred, true_masks).item()

            # compute the Precision and the Recall
            pre, rec = compute_pre_rec(masks_pred, true_masks, multi_class=multi_class)
            precision += pre
            recall += rec

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score, precision, recall
    else:
        return (
            dice_score / num_val_batches,
            precision / num_val_batches,
            recall / num_val_batches,
        )
