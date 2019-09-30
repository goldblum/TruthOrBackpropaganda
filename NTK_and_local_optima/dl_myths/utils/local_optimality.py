"""Stuff to check local optimality."""
import torch
import warnings
import time


def gradient_norm(dataloader, net, loss_fn, device, weight_decay=0):
    """Compute gradient over full data."""
    net.eval()
    net.zero_grad()
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
    grad_norm = 0
    for p in net.parameters():
        grad_norm += ((p.grad + p.data * weight_decay) / (i + 1)).pow(2).sum()

    net.zero_grad()
    return grad_norm.sqrt().detach()
