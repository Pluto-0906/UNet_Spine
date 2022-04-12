import torch
import torch.nn.functional as F


def focal_loss(input, target, alpha=0.25, gamma=2, size_average=True, num_classes=2):
    alpha_list = torch.zeros(num_classes)
    alpha_list[0] += alpha
    alpha_list[1:] += 1 - alpha
    alpha = alpha_list

    # input:(b,c,w,h) target(b,w,h)
    alpha = alpha.to(input.device)
    preds_softmax = F.softmax(input, dim=1)
    # input:(b,c,w,h) -> input(b*w*h,c)
    preds_softmax = preds_softmax.view(-1, preds_softmax.size(1))
    preds_logsoft = torch.log(preds_softmax)
    target = target.to(torch.int64)

    preds_softmax = preds_softmax.gather(1, target.view(-1, 1))
    preds_logsoft = preds_logsoft.gather(1, target.view(-1, 1))
    alpha = alpha.gather(0, target.view(-1))
    loss = -torch.mul(torch.pow((1 - preds_softmax), gamma), preds_logsoft)

    loss = torch.mul(alpha, loss.t())
    if size_average:
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss
