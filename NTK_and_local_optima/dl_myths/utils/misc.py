"""
 Other checks
"""
import torch
from collections import defaultdict


def check_throughputs(net, dataloader, quiet=True, device=torch.device('cpu')):
    """Place a hook after at relu layer in the model and count the number of non-negative inputs."""
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    net_device = next(net.parameters()).device
    net.to(device)
    hooks = []
    num_negative = defaultdict(lambda: 0)

    def check_nonnegative(self, input, output):
        module_name = str(*[name for name, mod in net.named_modules() if self is mod])
        neg_outputs = torch.as_tensor(input[0].detach() < 0, device=device)
        num_negative[module_name] += neg_outputs.sum().item() / neg_outputs.numel()
        if not quiet:
            print(input[0].shape)
            print(f'Input Features of {module_name} analyzed. {neg_outputs.sum().item()} negatives.')

    for name, module in net.named_modules():
        if 'relu' in name.lower():
            hooks.append(module.register_forward_hook(check_nonnegative))

    try:
        for i, (inputs, _) in enumerate(dataloader):
            outputs = net(inputs.to(device))
    except Exception as e:
        print(repr(e))

    for hook in hooks:
        hook.remove()
    
    net.to(net_device)
    return {key: num / (i + 1) for key, num in num_negative.items()}
