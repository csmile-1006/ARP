import re

import torch


def get_activation(name, activation: dict):
    def hook(model, input, output):
        if isinstance(output, torch.Tensor):
            activation[name] = output.detach()

    return hook


def attach_intermediate_layer(model, activation: dict):
    for name, layer in model.named_modules():
        # if "" in name:
        if re.match(r"visual.transformer.resblocks.[0-9]*$", name) or re.match(r"transformer.resblocks.[0-9]*$", name):
            layer.register_forward_hook(get_activation(name, activation))


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
