from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class CnnBasicBlock(nn.Module):
    inchan: int = 16
    batch_norm: bool = False
    padding: str = "SAME"

    def setup(self):
        self.conv0 = nn.Conv(self.inchan, kernel_size=(3, 3), padding=self.padding)
        self.conv1 = nn.Conv(self.inchan, kernel_size=(3, 3), padding=self.padding)

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        x = observation
        if self.batch_norm:
            x = nn.BatchNorm()(x)
        x = nn.relu(x)
        x = self.conv0(x)
        if self.batch_norm:
            x = nn.BatchNorm()(x)
        x = nn.relu(x)
        x = self.conv1(x)

        return observation + x


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    outchan: int = 16
    pool: bool = True
    padding: str = "SAME"
    nblock: int = 2

    def setup(self):
        self.firstconv = nn.Conv(self.outchan, kernel_size=(3, 3), padding=self.padding)

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        x = self.firstconv(observation)
        if self.pool:
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=self.padding)
        for _ in range(self.nblock):
            x = CnnBasicBlock(self.outchan)(x)
        return x


class ImpalaCNN(nn.Module):
    chans: Sequence[int] = (16, 32, 32)
    outsize: int = 256
    padding: str = "SAME"
    final_relu: bool = True
    nblock: int = 2

    # input: scaled observations. (x / 255.0)
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # shape of x : (batch_size * timestep, h, w, c)
        for outchan in self.chans:
            x = CnnDownStack(outchan=outchan, nblock=self.nblock)(x)
        _, h, w, c = x.shape

        # flatten image
        x = x.reshape(-1, h * w * c)
        x = nn.relu(x)
        x = nn.Dense(self.outsize)(x)
        if self.final_relu:
            x = nn.relu(x)
        return x
