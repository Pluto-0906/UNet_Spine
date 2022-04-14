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
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        mask_true = mask_true.unsqueeze(dim=1)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = torch.sigmoid(mask_pred)

            # compute dice score
            dice_score += multiclass_dice_coeff(mask_pred, mask_true).item()

            # compute the Precision and the Recall
            pre, rec = compute_pre_rec(mask_pred, mask_true)
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
