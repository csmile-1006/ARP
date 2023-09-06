from typing import Callable, Optional

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np



class FeedForward(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, batch, deterministic=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        x = nn.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="fc1",
        )(batch)

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(
            self.out_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="fc2",
        )(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    use_bias: bool = False
    att_drop: float = 0
    proj_drop: float = 0
    kernel_init: Callable = nn.linear.default_kernel_init
    bias_init: Callable = nn.initializers.zeros
    deterministic: Optional[bool] = None
    alibi_bias: bool = False

    @nn.compact
    def __call__(self, batch, deterministic=None, custom_mask=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        qkv = nn.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(batch)
        qkv = jnp.split(qkv, 3, axis=-1)

        mh_fn = lambda x: einops.rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)
        q, k, v = jax.tree_map(mh_fn, qkv)

        scale = (self.dim // self.num_heads) ** -0.5
        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale

        n = attention.shape[-1]
        if self.alibi_bias:
            slopes = np.array(_get_attention_slopes(self.num_heads))
            pos_bias = slopes[:, None, None] * np.arange(n)[None, None, :]
            pos_bias = pos_bias[None, :, :, :]
            attention = attention + pos_bias

        mask = custom_mask
        if mask is None:
            mask = np.tril(np.ones((n, n)))[None, None, ...]
            mask = jnp.broadcast_to(mask, attention.shape)

        big_neg = jnp.finfo(attention.dtype).min
        attention = jnp.where(mask == 0, big_neg, attention)
        attention = nn.softmax(attention, axis=-1)
        attention = nn.Dropout(self.att_drop)(attention, deterministic)

        x = einops.rearrange(attention @ v, "b h n d -> b n (h d)")
        x = nn.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


def _get_attention_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(np.math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if np.math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** np.math.floor(np.math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + _get_attention_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


class Block(nn.Module):
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    att_drop: float = 0.0
    drop: float = 0.0
    alibi_bias: bool = False

    @nn.compact
    def __call__(self, batch, deterministic=False, custom_mask=None):
        x = nn.LayerNorm()(batch)
        x = Attention(
            self.dim,
            self.num_heads,
            True,
            self.att_drop,
            self.drop,
            alibi_bias=self.alibi_bias,
        )(x, deterministic, custom_mask)
        batch = batch + x

        x = nn.LayerNorm()(batch)
        x = FeedForward(self.dim * self.mlp_ratio, self.dim, self.drop)(x, deterministic)
        return batch + x


class Transformer(nn.Module):
    emb_dim: int = 1024
    depth: int = 24
    att_drop: float = 0
    drop: float = 0
    num_heads: int = 16
    mlp_ratio: int = 4
    alibi_bias: bool = False

    @nn.compact
    def __call__(self, x, deterministic=False, custom_mask=None):
        for _ in range(self.depth):
            x = Block(
                self.emb_dim,
                self.num_heads,
                self.mlp_ratio,
                self.att_drop,
                self.drop,
                self.alibi_bias,
            )(x, deterministic, custom_mask)

        x = nn.LayerNorm()(x)
        return x
