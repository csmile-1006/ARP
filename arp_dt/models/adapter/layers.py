from typing import Callable, Literal, Union

import flax.linen as nn


class AdapterMLP(nn.Module):
    hidden_dim: int = 1024
    output_dim: int = 1024
    num_layers: int = 2
    activation: str = "relu"
    weight_init: Callable = nn.initializers.xavier_uniform
    norm_type: Union[Literal["batchnorm", "layernorm"], None] = None

    def setup(self):
        if self.activation == "relu":
            self.activation_layer = nn.relu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_dim, kernel_init=self.weight_init())(x)
            if self.norm_type == "batchnorm":
                x = nn.BatchNorm(x)
            elif self.norm_type == "layernorm":
                x = nn.LayerNorm(x)
            x = self.activation_layer(x)

        x = nn.Dense(self.output_dim, kernel_init=self.weight_init())(x)
        x = self.activation_layer(x)
        return x
