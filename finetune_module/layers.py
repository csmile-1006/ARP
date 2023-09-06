from typing import Literal, Union

from torch import nn


class AdapterMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 1024,
        num_layers: int = 2,
        activation: str = "relu",
        weight_init: str = "orthogonal",
        bias_init: str = "zeros",
        norm_type: Union[Literal["batchnorm", "layernorm"], None] = None,
    ):
        super().__init__()

        if not norm_type:
            norm_type = nn.Identity
        elif norm_type == "batchnorm":
            norm_type = nn.BatchNorm1d
        elif norm_type == "layernorm":
            norm_type = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported norm layer: {norm_type}")

        if activation == "relu":
            act_layer = nn.ReLU

        if weight_init == "orthogonal":
            try:
                gain = nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0
            weight_init = lambda x: nn.init.orthogonal_(x, gain=gain)

        if bias_init == "zeros":
            bias_init = nn.init.zeros_

        hidden_depth = num_layers - 1
        mods = [nn.Linear(input_dim, hidden_dim), norm_type(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_type(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))

        for mod in mods:
            if isinstance(mod, nn.Linear):
                weight_init(mod.weight)
                bias_init(mod.bias)

        self.layers = nn.Sequential(*mods)

    def forward(self, x):
        return self.layers(x)
