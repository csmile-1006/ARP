
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import transformers
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict

from .layers import Transformer
from .models.adapter.layers import AdapterMLP
from .models.m3ae import model as m3ae
from .models.openai import model as clip
from .utils import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed


class BC(nn.Module):
    config_updates: ... = None
    num_actions: int = None
    patch_dim: int = None
    normalize_quterion: bool = True

    # fmt: off
    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.model_type = config_dict.placeholder(str)
        config.transfer_type = "none"
        config.alibi_bias = False
        config.att_drop = 0.0
        config.drop = 0.0
        config.mlp_ratio = 4
        config.emb_dim = 128
        config.depth = 2
        config.num_heads = 8
        config.use_discrete_action = False
        config.vox_size = 100
        config.rotation_resolution = 5
        config.scene_bound = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]

        # custom setup for baseline
        # use another type of CLIP (scratch, pre-trained one (video2text))
        config.use_adapter = False
        config.use_from_scratch = False
        config.use_impala_backbone = False
        config.clip_checkpoint_path = "none"

        config.use_intermediate = False
        config.num_ensembles = 5

        config.mae = m3ae.MaskedAutoencoder.get_default_config()
        config.m3ae = m3ae.MaskedMultimodalAutoencoder.get_default_config()
        if config.model_type is not None:
            get_transformer_by_config(config.model_type, config)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config
    # fmt: on

    @nn.nowrap
    def rng_keys(self):
        self._rng_keys = ["params", "noise", "dropout"]
        return self._rng_keys

    @nn.nowrap
    def no_decay_list(self):
        no_decay = []
        return no_decay

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

        self.policy = Transformer(
            emb_dim=self.config.emb_dim,
            depth=self.config.depth,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            alibi_bias=self.config.alibi_bias,
        )

        self.action_outputs = [
            nn.Sequential([nn.Dense(self.config.emb_dim), nn.relu, nn.Dense(self.num_actions, use_bias=False)])
        ] * self.config.num_ensembles
        # self.action_output = nn.Dense(self.num_actions, use_bias=False)

        if self.config.use_discrete_action:
            assert self.num_actions == 15, "15 discrete actions for Procgen benchmark."
            self.action_input = nn.Embed(num_embeddings=self.num_actions, features=self.config.emb_dim)
        else:
            self.action_input = nn.Dense(self.config.emb_dim, use_bias=False)

        self.state_input = nn.Dense(self.config.emb_dim, use_bias=False)

        self.patchify = lambda x: einops.rearrange(
            x,
            "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
            p1=self.patch_dim,
            p2=self.patch_dim,
        )
        transfer_type = self.config.transfer_type
        if transfer_type == "none":
            self.patch_emb = nn.Dense(self.config.emb_dim)
        elif transfer_type.startswith("clip"):
            model_name = transfer_type.split("_", 1)[1]
            self.pt_model = clip.MODELS[model_name]()
            self.pt_params = clip.load_model_vars(model_name)
            self.image_text_input = nn.Dense(self.config.emb_dim)
        elif transfer_type.startswith("m3ae"):
            model_name = transfer_type.split("_", 1)[1]
            text_vocab_size = transformers.BertTokenizer.from_pretrained("bert-base-uncased").vocab_size
            self.pt_model = m3ae.MaskedMultimodalAutoencoder(self.config.m3ae, text_vocab_size=text_vocab_size)
            self.pt_params = m3ae.load_m3ae_model_vars(model_name)
            self.image_text_input = nn.Dense(self.config.emb_dim)
        else:
            raise ValueError("Unsupported transfer type!")

        if self.config.use_adapter:
            self.residual_weight = self.param(
                "residual_weight",
                nn.initializers.constant(4.0),
                (1),
            )

    @nn.compact
    def __call__(self, batch, deterministic=False):
        batch_size, num_timestep = batch["action"].shape[:2]

        num_obs_token, image_embed, action_embed, state_embed = self.encode(batch)

        if state_embed is not None:
            token_embed = jnp.concatenate([image_embed, state_embed, action_embed], axis=-1)
            num_token_per_step = num_obs_token + 1 + 1
        else:
            token_embed = jnp.concatenate([image_embed, action_embed], axis=-1)
            num_token_per_step = num_obs_token + 1

        token_embed = jnp.reshape(
            token_embed,
            [batch_size, num_token_per_step * num_timestep, self.config.emb_dim],
        )

        custom_mask = None
        if self.config.model_type.startswith("vit"):
            seq_len = token_embed.shape[1]
            causal_mask = np.tril(np.ones((seq_len, seq_len)))
            num_non_obs_tokens = num_token_per_step - num_obs_token
            diag = [
                np.ones((num_obs_token, num_obs_token))
                if i % 2 == 0
                else np.zeros((num_non_obs_tokens, num_non_obs_tokens))
                for i in range(num_timestep * 2)
            ]
            block_diag = scipy.linalg.block_diag(*diag)
            custom_mask = np.logical_or(causal_mask, block_diag)
            custom_mask = custom_mask.astype(np.float64)[None, None, ...]

        output_embed = self.policy(token_embed, deterministic=deterministic, custom_mask=custom_mask)
        if state_embed is None:
            action_pred = output_embed[:, (num_obs_token - 1) :: num_token_per_step, :]
        else:
            action_pred = output_embed[:, num_obs_token::num_token_per_step, :]
        action_preds = []
        for en_idx in range(self.config.num_ensembles):
            _action_pred = self.action_outputs[en_idx](action_pred)
            action_preds.append(_action_pred)

        action_pred = jnp.stack(action_preds).mean(axis=0)
        # action_pred = self.action_output(action_pred)
        # if self.num_actions > 4:
        #     action_pred = self.postprocess_action(action_pred)

        loss, acc = self.compute_loss(action_pred, batch["action"])

        output = {"action_pred": action_pred, "loss": loss, "acc": acc}
        return output

    def postprocess_action(self, action_pred):
        if not self.config.use_discrete_action:
            if self.normalize_quterion:  # normalize quterion
                x = action_pred[..., 3:7]
                x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
                action_pred = action_pred.at[:, 3:7].set(x)

    def compute_loss(self, action_pred, action):
        if not self.config.use_discrete_action:
            if self.normalize_quterion:  # normalize quterion
                x = action_pred[:, 3:7]
                x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
                action_pred = action_pred.at[:, 3:7].set(x)

            loss = mse_loss(action_pred, action)
            return loss, 0.0
        else:
            # action_pred shape: (batch_size, num_timestep, action_dim)
            # action shape: (batch_size, num_timestep)
            trans_loss = cross_entropy(logits=action_pred, labels=action, num_classes=self.num_actions)
            loss, acc = trans_loss
            return loss, acc

    def encode(self, batch):
        text = batch.get("instruct", None)

        image_batch = batch["image"]
        image = jnp.array(list(image_batch.values()))
        # image = (image / 255.0).astype(np.float32)
        num_image, batch_size, num_timestep = image.shape[:3]

        state_batch = batch.get("state", None)
        if state_batch is not None:
            state_emb = self.state_input(state_batch)
        else:
            state_emb = None

        action_batch = batch["action"]
        action_emb = self.action_input(action_batch)

        text_padding_mask = batch.get("text_padding_mask", None)

        transfer_type = self.config.transfer_type

        def concat_multiple_image_emb(img_emb):
            # input of img_emb: (num_iamge, batch_size,  num_timestep, emb_dim)
            img_emb = jnp.reshape(img_emb, (batch_size * num_image, num_timestep, -1))
            img_emb = jnp.concatenate(
                jnp.split(img_emb, num_image, axis=0), -1
            )  # output of img_emb: (batch_size, num_timestep, num_image * emb_dim)
            return img_emb

        if transfer_type == "none":
            image = jnp.concatenate(
                list(image_batch.values()), axis=-1
            )  # (batch_size, num_timestep, image.shape[-3:-1], image.shape[-1] * num_image)
            image = jnp.reshape(
                image, (-1,) + image.shape[-3:]
            )  # (batch_size * num_timestep, image.shape[-3:-1], image.shape[-1] * num_image)

            patch = self.patch_emb(self.patchify(image))

            num_obs_token = patch.shape[1]

            patch = patch + get_2d_sincos_pos_embed(patch.shape[-1], num_obs_token)  # add 2d positional embedding
            patch = jnp.reshape(patch, (batch_size, num_timestep, -1))
            patch = patch + get_1d_sincos_pos_embed(patch.shape[-1], num_timestep)  # add 1d positional embedding

            return num_obs_token, patch, action_emb, state_emb

        elif transfer_type.startswith("clip"):
            image = jnp.reshape(
                image, (-1,) + image.shape[-3:]
            )  # (batch_size * num_image * num_timestep, image.shape[-3:])

            image = jax.image.resize(
                image, (image.shape[0], 224, 224, image.shape[-1]), method="bicubic"
            )  # to meet the input size of the clip model
            img_emb = self.pt_model.apply(self.pt_params, image, method=self.pt_model.encode_image)

            img_emb = concat_multiple_image_emb(img_emb)

            if text is not None:
                text_emb = self.pt_model.apply(self.pt_params, text, method=self.pt_model.encode_text)
                text_emb = jnp.tile(
                    jnp.expand_dims(text_emb, axis=0),
                    (img_emb.shape[0], img_emb.shape[1], 1),
                )

                image_text_emb = jnp.concatenate([img_emb, text_emb], axis=-1)
            else:
                image_text_emb = img_emb

            image_text_emb = jax.lax.stop_gradient(image_text_emb)
            image_text_emb = nn.tanh(self.image_text_input(image_text_emb))
            image_text_emb = image_text_emb + get_1d_sincos_pos_embed(image_text_emb.shape[-1], num_timestep)

            return 1, image_text_emb, action_emb, state_emb

        elif transfer_type.startswith("m3ae"):
            image = jnp.reshape(
                image, (-1,) + image.shape[-3:]
            )  # (batch_size * num_image * num_timestep, image.shape[-3:])

            patch = self.patchify(image)
            if text is not None:
                tokenized_caption = jnp.tile(text, (num_image * num_timestep, 1))
                text_padding_mask = jnp.tile(text_padding_mask, (num_image * num_timestep, 1))
            else:
                tokenized_caption = None
                text_padding_mask = None

            if self.config.use_intermediate:
                image_text_emb, states = self.pt_model.apply(
                    self.pt_params,
                    patch,
                    tokenized_caption,
                    text_padding_mask,
                    method=self.pt_model.forward_representation,
                    deterministic=True,
                    capture_intermediates=True,
                    mutable=["intermediates"],
                )
                num_layers = self.config.m3ae.depth
                intermediate_embs = [
                    states["intermediates"]["encoder"][f"intermediate_layer_{i}"][0] for i in range(num_layers - 1)
                ]
                image_text_emb = jnp.concatenate(intermediate_embs + [image_text_emb], axis=0)
            else:
                image_text_emb = self.pt_model.apply(
                    self.pt_params,
                    patch,
                    tokenized_caption,
                    text_padding_mask,
                    method=self.pt_model.forward_representation,
                    deterministic=True,
                )
                num_layers = 1

            # image_text_emb shape: (num_layers * num_image * batch_size * num_timestep, num_patches, emb_dim)
            image_text_emb = jax.lax.stop_gradient(image_text_emb)

            # image_text_emb shape: (num_layers * num_image * batch_size * num_timestep, num_patches, emb_dim)
            if self.config.use_adapter:
                adapter = AdapterMLP(
                    hidden_dim=image_text_emb.shape[-1], output_dim=image_text_emb.shape[-1], num_layers=2
                )
                adapter_image_text_emb = adapter(image_text_emb)
                res = nn.sigmoid(self.residual_weight)
                image_text_emb = res * adapter_image_text_emb + (1 - res) * image_text_emb

            # image_text_emb shape: (num_layers * num_image * batch_size * num_timestep, num_patches * emb_dim)
            image_text_emb = jnp.reshape(image_text_emb, (batch_size * num_image * num_layers, num_timestep, -1))
            # image_text_emb = jnp.mean(image_text_emb, axis=-2)
            # image_text_emb = jnp.reshape(image_text_emb, (batch_size * num_image * num_layers, num_timestep, -1))

            # image_text_emb shape: (batch_size * num_image, num_timestep, num_layers * emb_dim)
            image_text_emb = jnp.concatenate(jnp.split(image_text_emb, num_layers, axis=0), -1)

            # image_text_emb shape: (batch_size * num_image, num_timestep, last_emb_dim)
            image_text_emb = nn.tanh(self.image_text_input(image_text_emb))

            # image_text_emb shape: (batch_size, num_timestep, num_image * last_emb_dim)
            image_text_emb = jnp.concatenate(jnp.split(image_text_emb, num_image, axis=0), -1)

            return num_image, image_text_emb, action_emb, state_emb

    def greedy_action(self, batch):
        if not self.config.use_discrete_action:
            return self(batch, deterministic=True)["action_pred"][:, -1, :]
        else:
            return self(batch, deterministic=True)["action_pred"][:, -1, :].argmax(-1)


def cross_entropy(logits, labels, num_classes):
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    labels = jax.nn.one_hot(labels, num_classes)
    loss = jnp.mean(-labels * jax.nn.log_softmax(logits))
    return loss, acc


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


def get_pos_embed(embed_dim, length):
    assert embed_dim % 2 == 0
    pos = jnp.arange(length, dtype=jnp.float32)
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return jnp.expand_dims(emb, 0)


def get_transformer_by_config(model_type, config):
    if model_type.startswith("tiny"):
        if model_type == "tiny":
            config.emb_dim = 128
        elif model_type == "tinyl":
            config.emb_dim = 2560
        elif model_type == "tinyxl":
            config.emb_dim = 5120
        elif model_type == "tinyxxl":
            config.emb_dim = 398592
        else:
            raise ValueError("unknown model type {}".format(model_type))
        config.depth = 4
        config.num_heads = 8
    elif model_type.startswith("small"):
        if model_type == "small":
            config.emb_dim = 512
        elif model_type == "smalll":
            config.emb_dim = 2560
        elif model_type == "smallxl":
            config.emb_dim = 5120
        elif model_type == "smallxxl":
            config.emb_dim = 398592
        else:
            raise ValueError("unknown model type {}".format(model_type))
        config.depth = 4
        config.num_heads = 8
    elif model_type.startswith("base"):
        if model_type == "base":
            config.emb_dim = 768
        elif model_type == "basel":
            config.emb_dim = 2560
        elif model_type == "basexl":
            config.emb_dim = 5120
        elif model_type == "basexxl":
            config.emb_dim = 398592
        else:
            raise ValueError("unknown model type {}".format(model_type))
        config.depth = 6
        config.num_heads = 12
    elif model_type.startswith("medium"):
        if model_type == "medium":
            config.emb_dim = 1280
        elif model_type == "mediuml":
            config.emb_dim = 2560
        elif model_type == "mediumxl":
            config.emb_dim = 5120
        elif model_type == "mediumxxl":
            config.emb_dim = 398592
        else:
            raise ValueError("unknown model type {}".format(model_type))
        config.depth = 10
        config.num_heads = 20
    elif model_type.startswith("large"):
        if model_type == "large":
            config.emb_dim = 1280
        elif model_type == "largel":
            config.emb_dim = 2560
        elif model_type == "largexl":
            config.emb_dim = 5120
        elif model_type == "largexxl":
            config.emb_dim = 398592
        else:
            raise ValueError("unknown model type {}".format(model_type))
        config.depth = 14
        config.num_heads = 20
    elif model_type.startswith("huge"):
        if model_type == "huge":
            config.emb_dim = 1280
        elif model_type == "hugel":
            config.emb_dim = 2560
        elif model_type == "hugexl":
            config.emb_dim = 5120
        elif model_type == "hugexxl":
            config.emb_dim = 398592
        else:
            raise ValueError("unknown model type {}".format(model_type))
        config.depth = 18
        config.num_heads = 16
    elif model_type == "debug":
        config.emb_dim = 16
        config.depth = 2
        config.num_heads = 2
        config.mlp_ratio = 2
    else:
        raise ValueError("Unsupported model type!")
