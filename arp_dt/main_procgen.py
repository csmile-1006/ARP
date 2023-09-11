import copy
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import pprint
from functools import partial
from typing import Sequence

import absl.app
import absl.flags
import augmax
import clip
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import torch
import wandb
from absl import app, logging
from flax import jax_utils
from flax.jax_utils import prefetch_to_device
from flax.training import common_utils
from flax.training.train_state import TrainState
from tqdm.auto import tqdm, trange

from .ARPDT import ARPDT
from .BC import BC
from .data_procgen import ProcgenDataset, get_clip_instruct, get_clip_special_instruct, get_m3ae_instruct
from .envs import rollout_procgen
from .envs.procgen import Procgen
from .GCBC import GCBC
from .utils import (
    JaxRNG,
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    load_pickle,
    next_rng,
    set_random_seed,
)

FLAGS_DEF = define_flags_with_default(
    seed=42,
    epochs=100,
    warmup_epochs=5.0,
    weight_decay=1e-4,
    batch_size=2,
    dataloader_n_workers=0,
    dataloader_shuffle=True,
    log_freq=100,
    save_model_freq=0,
    load_checkpoint="",
    lr=0.1,
    lr_schedule="cos",
    momentum=0.9,
    clip_gradient=1e9,
    auto_scale_lr=False,
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
    model=ARPDT.get_default_config(),
    data=ProcgenDataset.get_default_config(),
    env=Procgen.get_default_config(),
    window_size=4,
    use_text=False,
    val_every_epochs=10,
    test_every_epochs=10,
    num_test_episodes=5,
    return_to_go=0.0,
    scale=10.0,
    game_name="coinrun",
    is_tpu=False,
    use_vl=True,
    vl_type="clip",
    vl_checkpoint="",
    use_crop=True,
    eval_data_path="",
    eval_with_goal=False,
)
FLAGS = absl.flags.FLAGS


def build_env_fn(game_name, conf):
    def env_fn():
        env = Procgen(game_name, conf)
        return env

    return env_fn


@partial(jax.pmap, axis_name="pmap", donate_argnums=0)
def sync_state_fn(state):
    i = jax.lax.axis_index("pmap")

    def select(x):
        return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

    return jax.tree_map(select, state)


def create_train_step(model, learning_rate, weight_decay):
    def loss_fn(params, batch, rng):
        rng_generator = JaxRNG(rng)
        output = model.apply(
            {"params": params},
            batch,
            rngs=rng_generator(model.rng_keys()),
            deterministic=False,
        )
        loss = output["loss"]
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        aux = dict(
            loss=loss,
            acc=output["acc"] * 100,
            trans_loss=output.get("trans_loss", 0.0),
            return_loss=output.get("return_loss", 0.0),
            weight_penalty=weight_penalty,
            weight_l2=weight_l2,
        )
        return loss, (aux,)

    @partial(jax.pmap, axis_name="pmap", donate_argnums=(0))
    def train_step_fn(state, batch, rng):
        next_rng, split_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (aux,)), grads = jax.lax.pmean(grad_fn(state.params, batch, split_rng), axis_name="pmap")

        aux["train_state_step"] = state.step
        aux["learning_rate"] = learning_rate(state.step)

        new_state = state.apply_gradients(grads=grads)

        return new_state, aux, next_rng

    return train_step_fn


def create_val_step(model):
    def val_fn(params, batch, rng):
        rng_generator = JaxRNG(rng)
        output = model.apply(
            {"params": params},
            batch,
            rngs=rng_generator(model.rng_keys()),
            deterministic=True,
        )
        aux = dict(
            loss=output["loss"],
            trans_loss=output.get("trans_loss", 0.0),
            return_loss=output.get("return_loss", 0.0),
            acc=output["acc"] * 100,
        )

        return aux

    @partial(jax.pmap, axis_name="pmap")
    def val_step_fn(state, batch, rng):
        next_rng, split_rng = jax.random.split(rng)
        aux = jax.lax.pmean(val_fn(state.params, batch, split_rng), axis_name="pmap")
        return aux, next_rng

    return val_step_fn


def create_test_step(
    model,
    environment,
    episode_length,
    instruct,
    window_size,
    num_episodes,
    return_to_go,
    scale,
    transform_obs_fn,
    transform_action_fn,
    clip_model,
    vl_type,
    text,
    reward_min,
    use_normalize,
    use_crop,
    eval_data_path,
    data_name,
):
    @jax.jit
    def policy_fn(variables, inputs, rngs):
        inputs.update(instruct)
        output = model.apply(
            variables=variables,
            batch=inputs,
            rngs=rngs,
            method=model.greedy_action,
        )
        return output

    def test_step_fn(state, rng):
        next_rng, split_rng = jax.random.split(rng)
        rng_generator = JaxRNG(split_rng)
        policy = partial(policy_fn, variables={"params": state.params})
        metric, info, videos = rollout_procgen.batch_rollout(
            rng=rng_generator(model.rng_keys()),
            data_aug_rng=rng_generator(),
            env=environment,
            policy_fn=policy,
            transform_obs_fn=transform_obs_fn,
            transform_action_fn=transform_action_fn,
            episode_length=episode_length,
            window_size=window_size,
            num_episodes=num_episodes,
            return_to_go=return_to_go,
            scale=scale,
            clip_model=clip_model,
            vl_type=vl_type,
            text=text,
            reward_min=reward_min,
            use_normalize=use_normalize,
            use_crop=use_crop,
            eval_data_path=eval_data_path,
            data_name=data_name,
        )
        return metric, info, videos, next_rng

    return test_step_fn


def image_aug(augs: Sequence[str]):
    augs = augs.split(", ")
    if FLAGS.model.transfer_type.startswith("clip"):
        image_size = 224
    elif FLAGS.model.transfer_type.startswith("m3ae"):
        image_size = 256
    elif FLAGS.model.transfer_type.startswith("mae"):
        image_size = 256

    _transforms = [augmax.Resize(image_size, image_size), augmax.ByteToFloat()]

    for aug in augs:
        if aug == "random_crop":
            _transforms.extend(
                [
                    augmax.RandomCrop(
                        FLAGS.data.image_size * 0.8,
                        FLAGS.data.image_size * 0.8,
                    ),
                    augmax.Resize(image_size, image_size),
                ]
            )
        elif aug == "color_jitter":
            _transforms.append(augmax.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))
        elif aug == "rotate":
            _transforms.append(augmax.Rotate())

    _transforms.append(
        augmax.Normalize(mean=jnp.array([0.5762, 0.5503, 0.5213]), std=jnp.array([0.3207, 0.3169, 0.3307]))
    )

    def single_image_aug_fn(image, rng):
        transform = augmax.Chain(*_transforms)
        return transform(rng, image)

    single_image_aug_vmap_fn = jax.jit(jax.vmap(single_image_aug_fn))

    @partial(jax.pmap, axis_name="pmap")
    def multi_image_aug_fn(images, rng):
        num_rngs = images.shape[0]
        sub_rngs = jax.random.split(rng, num_rngs + 1)
        sub_rngs, new_rng = sub_rngs[:-1], sub_rngs[-1]
        return single_image_aug_vmap_fn(images, sub_rngs), new_rng

    return multi_image_aug_fn


@jax.jit
def test_image_aug(image, rng):
    next_rng, split_rng = jax.random.split(rng)
    if FLAGS.model.transfer_type.startswith("clip"):
        image_size = 224
    elif FLAGS.model.transfer_type.startswith("m3ae"):
        image_size = 256
    elif FLAGS.model.transfer_type.startswith("mae"):
        image_size = 256
    transform = augmax.Chain(
        augmax.Resize(width=image_size, height=image_size),
        augmax.CenterCrop(width=image_size, height=image_size),
        augmax.ByteToFloat(),
        augmax.Normalize(mean=jnp.array([0.5762, 0.5503, 0.5213]), std=jnp.array([0.3207, 0.3169, 0.3307])),
    )
    return transform(split_rng, image), next_rng


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    assert FLAGS.batch_size % jax_process_count == 0
    variant["process_batch_size"] = process_batch_size = FLAGS.batch_size // jax_process_count
    variant["device_batch_size"] = device_batch_size = process_batch_size // jax.local_device_count()
    if FLAGS.auto_scale_lr:
        lr_scale = FLAGS.batch_size / 256
    else:
        lr_scale = 1.0
    variant["effective_lr"] = FLAGS.lr * lr_scale
    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert process_batch_size % n_devices == 0

    FLAGS.logging.experiment_name = "-".join(
        [FLAGS.game_name, FLAGS.env.eval_env_type, FLAGS.env.distribution_mode, FLAGS.logging.notes]
    )
    # First setting for use discrete action in Procgen Benchmark.
    FLAGS.model.use_discrete_action = True
    if not FLAGS.use_vl and FLAGS.vl_type == "BC":
        # If not use clip, baseline would be InstructRL with text representation.
        FLAGS.use_text = True
        variant["use_text"] = True

    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
    )
    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    dataset_name = f"{FLAGS.game_name}_{FLAGS.env.distribution_mode}_level{FLAGS.env.start_level}to{FLAGS.env.num_levels}_num{FLAGS.data.num_demonstrations}_frame{FLAGS.data.num_frames}"
    if not FLAGS.data.enable_filter:
        dataset_name += "_unfiltered"
    if FLAGS.data.train_env_type != "none":
        dataset_name += f"_{FLAGS.data.train_env_type}"

    train_data_path = os.path.join(FLAGS.data.path, dataset_name)
    if FLAGS.eval_with_goal:
        test_data_path = os.path.join(
            FLAGS.data.path,
            f"{FLAGS.game_name}_{FLAGS.env.distribution_mode}_level{FLAGS.env.start_level+FLAGS.env.num_levels}to{FLAGS.env.num_levels*2}_num{FLAGS.num_test_episodes * 10}_frame{FLAGS.data.num_frames}",
        )
        if FLAGS.env.eval_env_type != "none":
            test_data_path += f"_{FLAGS.env.eval_env_type}"
    else:
        test_data_path = None

    train_dataset = ProcgenDataset(
        update=FLAGS.data,
        dataset_name=dataset_name,
        start_offset_ratio=jax_process_index / jax_process_count,
        split="train",
    )
    if FLAGS.use_vl:
        wandb.config.update(
            {
                "return_to_go": train_dataset.return_to_go,
                "scale": train_dataset.scale,
                "data.scale": train_dataset.config.scale,
            },
            allow_val_change=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=process_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )
    val_dataset = ProcgenDataset(
        update=FLAGS.data,
        dataset_name=dataset_name,
        start_offset_ratio=jax_process_index / jax_process_count,
        split="val",
    )
    val_batch_size = min(process_batch_size, len(val_dataset) // jax_process_count)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=FLAGS.dataloader_shuffle,
        drop_last=True,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )

    steps_per_epoch = int(len(train_dataset) / FLAGS.batch_size)
    total_steps = steps_per_epoch * FLAGS.epochs
    val_steps = int(len(val_dataset) / val_batch_size)

    if FLAGS.save_model_freq > 0:
        save_model_freq = FLAGS.save_model_freq
    else:
        save_model_freq = steps_per_epoch * FLAGS.test_every_epochs

    normalize_quterion = False
    if FLAGS.use_vl or FLAGS.data.use_task_reward:
        model = ARPDT(
            config_updates=FLAGS.model,
            num_actions=train_dataset.num_actions,
            patch_dim=16,
            normalize_quterion=normalize_quterion,
        )
    elif "GCBC" in FLAGS.vl_type:
        model = GCBC(
            config_updates=FLAGS.model,
            num_actions=train_dataset.num_actions,
            patch_dim=16,
            normalize_quterion=normalize_quterion,
        )
    else:
        model = BC(
            config_updates=FLAGS.model,
            num_actions=train_dataset.num_actions,
            patch_dim=16,
            normalize_quterion=normalize_quterion,
        )

    if FLAGS.lr_schedule == "fixed":
        learning_rate = optax.linear_schedule(init_value=FLAGS.lr, end_value=FLAGS.lr, transition_steps=total_steps)
    elif FLAGS.lr_schedule == "cos":
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=FLAGS.lr * lr_scale,
            warmup_steps=int(FLAGS.warmup_epochs * steps_per_epoch),
            decay_steps=total_steps,
            end_value=0.0,
        )
    elif FLAGS.lr_schedule == "cos_decay":
        learning_rate = optax.cosine_decay_schedule(init_value=FLAGS.lr, decay_steps=total_steps)
    else:
        raise ValueError("Unsupported lr schedule!")

    def get_dummy_input():
        dummy_input = {
            "action": jnp.ones((1, FLAGS.window_size), dtype=jnp.int32),
        }
        if train_dataset.config.state_key != "":
            dummy_input["state"] = jnp.ones((1, FLAGS.window_size, train_dataset.config.state_dim), dtype=jnp.float32)
        dummy_input["image"] = {}
        dummy_input["goal"] = {}
        for k, v in train_dataset.obs_shape["image"].items():
            if FLAGS.model.transfer_type.startswith("clip"):
                image_size = 224
            elif FLAGS.model.transfer_type.startswith("m3ae"):
                image_size = 256
            elif FLAGS.model.transfer_type.startswith("mae"):
                image_size = 256
            dummy_input["image"][k] = jnp.ones((1, FLAGS.window_size, image_size, image_size, 3), dtype=jnp.float32)
            dummy_input["goal"][k] = jnp.ones((1, FLAGS.window_size, image_size, image_size, 3), dtype=jnp.float32)

        dummy_input["rtg"] = {}
        for k, v in train_dataset.obs_shape["rtg"].items():
            dummy_input["rtg"][k] = jnp.ones((1, FLAGS.window_size, *v), dtype=jnp.float32)

        if FLAGS.use_text:
            dummy_instruct = {
                "instruct": jnp.zeros((1, FLAGS.data.tokenizer_max_length), dtype=jnp.int32),
                "text_padding_mask": jnp.ones((1, FLAGS.data.tokenizer_max_length), dtype=jnp.int32),
            }
        else:
            dummy_instruct = {"instruct": None, "text_padding_mask": None}
        dummy_input.update(dummy_instruct)

        return dummy_input

    if FLAGS.load_checkpoint != "":
        checkpoint_data = load_pickle(FLAGS.load_checkpoint)
        state = jax_utils.replicate(checkpoint_data["state"], jax_devices)
        start_step = checkpoint_data["step"]
    else:

        @jax.jit
        def init(*args, **kwargs):
            return model.init(*args, **kwargs)

        dummy_input = get_dummy_input()
        params = init(next_rng(model.rng_keys()), dummy_input, deterministic=False)["params"]
        params = flax.core.frozen_dict.unfreeze(params)

        def get_weight_decay_mask(params):
            flattened_params = flax.traverse_util.flatten_dict(flax.core.frozen_dict.unfreeze(params))

            def decay(key):
                return any([nd in k for nd in model.no_decay_list() for k in key])

            return flax.traverse_util.unflatten_dict({key: decay(key) for key in flattened_params.keys()})

        tx = optax.chain(
            optax.clip_by_global_norm(FLAGS.clip_gradient),
            optax.adamw(
                learning_rate=learning_rate,
                weight_decay=FLAGS.weight_decay,
                b1=0.9,
                b2=0.999,
                mask=get_weight_decay_mask,
            ),
        )

        state = jax_utils.replicate(
            TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx,
            ),
            jax_devices,
        )
        start_step = 0

    def flops(params):
        f = lambda x: model.apply({"params": flax.core.freeze(params)}, x)
        xla_f = jax.xla_computation(f)
        dummy_input = get_dummy_input()
        computation = xla_f(dummy_input)
        module = computation.as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
        return analysis

    if jax.process_index() == 0:
        analysis = flops(jax_utils.unreplicate(state.params))
        logging.info(f"flops: {analysis['flops']}")
        logger.log({"cost/flops": analysis["flops"]})
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(jax_utils.unreplicate(state.params)))
        logging.info(f"num_params: {num_params}")
        logger.log({"cost/num_params": num_params})

    train_step_fn = create_train_step(model, learning_rate, FLAGS.weight_decay)
    val_step_fn = create_val_step(model)
    if not FLAGS.is_tpu:
        train_env_conf = copy.deepcopy(FLAGS.env)
        train_env_conf.env_type = FLAGS.data.train_env_type
        train_env_conf.use_train_levels = True
        train_environment = build_env_fn(FLAGS.game_name, train_env_conf)()
        test_environment = build_env_fn(FLAGS.game_name, FLAGS.env)()

        if FLAGS.use_text:
            tokenizer = train_dataset.build_tokenizer()
            test_instruct, test_padding_mask = tokenizer(get_m3ae_instruct(FLAGS.game_name))
            test_instruct = test_instruct[None, ...]
        else:
            test_instruct = None
            test_padding_mask = None
        instruct_info = {"instruct": test_instruct, "text_padding_mask": test_padding_mask}

        transform_action_fn = lambda x: x

        clip_model, preprocess, text = None, None, None
        if FLAGS.use_vl:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            game_name = (
                FLAGS.game_name if FLAGS.env.eval_env_type == "none" else f"{FLAGS.game_name}_{FLAGS.env.eval_env_type}"
            )
            if FLAGS.data.inst_type != "none":
                text = get_clip_special_instruct(game_name, FLAGS.data.inst_type)
            else:
                text = get_clip_instruct(game_name)
            print(f"text: {text}")

            if FLAGS.vl_type == "clip" or FLAGS.vl_type == "clip_goal_conditioned":
                clip_model, preprocess = clip.load("ViT-B/16", device=device)
            elif FLAGS.vl_type.startswith("clip_"):
                _, preprocess = clip.load("ViT-B/16", device=device)
                if FLAGS.vl_type == "clip_ft":
                    print("use fine_tuned CLIP.")
                    from finetune_module.clip_multiscale_adapter import CLIPMultiscaleAdapter

                    use_id_loss = True if "vip_id" in FLAGS.vl_type else False
                    clip_model = CLIPMultiscaleAdapter(
                        device=device,
                        use_discrete_action=True,
                        action_dim=train_dataset.num_actions,
                        use_id_loss=use_id_loss,
                    ).to(device)
                assert FLAGS.vl_checkpoint != "", "You have to specifiy vl_checkpoint."
                model_state_dict = torch.load(FLAGS.vl_checkpoint)
                clip_model.load_state_dict(model_state_dict, strict=False)
                clip_model.eval()
            assert clip_model is not None

        partial_test_step_fn = partial(
            create_test_step,
            model=model,
            episode_length=FLAGS.env.episode_length,
            instruct=instruct_info,
            window_size=FLAGS.window_size,
            return_to_go=getattr(train_dataset, "return_to_go", 1000.0)
            if FLAGS.return_to_go == 0
            else FLAGS.return_to_go,
            scale=getattr(train_dataset, "scale", 100.0),
            transform_obs_fn=test_image_aug,
            transform_action_fn=transform_action_fn,
            clip_model=(clip_model, preprocess),
            vl_type=FLAGS.vl_type,
            text=text,
            reward_min=getattr(train_dataset, "reward_min", 0.0),
            use_normalize=FLAGS.data.use_normalize,
            use_crop=FLAGS.use_crop,
        )

        train_env_test_step_fn = partial_test_step_fn(
            environment=train_environment,
            num_episodes=FLAGS.num_test_episodes,
            eval_data_path=train_data_path,
            data_name="data_train.hdf5",
        )
        test_env_test_step_fn = partial_test_step_fn(
            environment=test_environment,
            num_episodes=FLAGS.num_test_episodes,
            eval_data_path=test_data_path,
            data_name="data_test.hdf5",
        )
        final_train_env_test_step_fn = partial_test_step_fn(
            environment=train_environment,
            num_episodes=FLAGS.num_test_episodes * 10,
            eval_data_path=train_data_path,
            data_name="data_train.hdf5",
        )
        final_test_env_test_step_fn = partial_test_step_fn(
            environment=test_environment,
            num_episodes=FLAGS.num_test_episodes * 10,
            eval_data_path=test_data_path,
            data_name="data_test.hdf5",
        )

    state = sync_state_fn(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    image_aug_rng, image_aug_fn = None, None
    image_aug_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)
    image_aug_fn = image_aug(FLAGS.data.augmentations)

    def generate_batch(iterator, image_aug_rng=None, image_aug_fn=None):
        while True:
            for batch in iterator:
                reshape_fn = lambda x: x.numpy().reshape(n_devices, -1, *x.shape[1:])

                if image_aug_fn:
                    image = jax.tree_util.tree_map(
                        lambda x: x.numpy().reshape(n_devices, -1, *x.shape[2:]), batch["image"]
                    )
                    for key in image:
                        _val, image_aug_rng = image_aug_fn(image[key], image_aug_rng)
                        image[key] = _val

                    image = jax.tree_util.tree_map(
                        lambda x: x.reshape(n_devices, -1, FLAGS.window_size, *x.shape[2:]), image
                    )
                else:
                    image = jax.tree_util.tree_map(reshape_fn, batch["image"])

                if "GCBC" in FLAGS.vl_type:
                    if image_aug_fn:
                        goal = jax.tree_util.tree_map(
                            lambda x: x.numpy().reshape(n_devices, -1, *x.shape[2:]), batch["goal"]
                        )
                        for key in goal:
                            _val, image_aug_rng = image_aug_fn(goal[key], image_aug_rng)
                            goal[key] = _val

                        goal = jax.tree_util.tree_map(
                            lambda x: x.reshape(n_devices, -1, FLAGS.window_size, *x.shape[2:]), goal
                        )
                    else:
                        goal = jax.tree_util.tree_map(reshape_fn, batch["goal"])
                else:
                    goal = None

                if "state" in batch:
                    state = jax.tree_util.tree_map(reshape_fn, batch["state"])
                else:
                    state = None
                action = jax.tree_util.tree_map(reshape_fn, batch["action"])
                rtg = jax.tree_util.tree_map(reshape_fn, batch["rtg"])
                if batch["instruct"] is not None and FLAGS.use_text:
                    instruct = jax.tree_util.tree_map(reshape_fn, batch["instruct"])
                else:
                    instruct = None
                if batch["text_padding_mask"] is not None and FLAGS.use_text:
                    text_padding_mask = jax.tree_util.tree_map(reshape_fn, batch["text_padding_mask"])
                else:
                    text_padding_mask = None

                yield {
                    "image": image,
                    "goal": goal,
                    "state": state,
                    "action": action,
                    "rtg": rtg,
                    "instruct": instruct,
                    "text_padding_mask": text_padding_mask,
                }

    train_iter = prefetch_to_device(
        generate_batch(train_loader, image_aug_rng=image_aug_rng, image_aug_fn=image_aug_fn), 2, jax_devices
    )
    val_iter = prefetch_to_device(
        generate_batch(val_loader, image_aug_rng=image_aug_rng, image_aug_fn=image_aug_fn), 2, jax_devices
    )
    step_counter = trange(start_step, total_steps, desc="Train...", ncols=0)

    best_eval_score = 0.0
    for step, batch in zip(step_counter, train_iter):
        if step % steps_per_epoch == 0 or FLAGS.load_checkpoint != "":
            train_metrics = []

        epoch = step // steps_per_epoch

        state, metrics, sharded_rng = train_step_fn(state, batch, sharded_rng)
        train_metrics.append(metrics)

        if step and step % FLAGS.log_freq == 0:
            log_metrics = common_utils.get_metrics(train_metrics)
            log_metrics = {f"train_{k}": v for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()}
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.val_every_epochs > 0 and step > 0 and step % (FLAGS.val_every_epochs * steps_per_epoch) == 0:
            val_metrics = []
            for _, batch in zip(trange(val_steps, desc="val...", ncols=0), val_iter):
                metrics, _ = val_step_fn(state, batch, sharded_rng)
                val_metrics.append(metrics)

            log_metrics = common_utils.get_metrics(val_metrics)
            log_metrics = {f"val_{k}": v for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()}
            log_metrics.update({"step": step, "epoch": epoch})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if (
            not FLAGS.is_tpu
            and FLAGS.test_every_epochs > 0
            and step > 0
            and (step % (FLAGS.test_every_epochs * steps_per_epoch) == 0 or step == total_steps - 1)
        ):
            if step == total_steps - 1:
                del train_env_test_step_fn
                del test_env_test_step_fn
                train_log_metrics, train_log_infos, train_videos, _ = final_train_env_test_step_fn(
                    flax.jax_utils.unreplicate(state), next_rng()
                )
                test_log_metrics, test_log_infos, test_videos, _ = final_test_env_test_step_fn(
                    flax.jax_utils.unreplicate(state), next_rng()
                )
            else:
                train_log_metrics, train_log_infos, _, _ = train_env_test_step_fn(
                    flax.jax_utils.unreplicate(state), next_rng()
                )
                test_log_metrics, test_log_infos, _, _ = test_env_test_step_fn(
                    flax.jax_utils.unreplicate(state), next_rng()
                )

            train_log_metrics = {
                f"test/train_{k}": v for k, v in jax.tree_map(lambda x: jax.device_get(x)[0], train_log_metrics).items()
            }
            train_log_metrics.update({"step": step, "epoch": epoch})
            logger.log(train_log_metrics)
            test_log_metrics = {
                f"test/test_{k}": v for k, v in jax.tree_map(lambda x: jax.device_get(x)[0], test_log_metrics).items()
            }
            test_log_metrics.update({"step": step, "epoch": epoch})
            logger.log(test_log_metrics)

            if train_log_infos["vid"] is not None and test_log_infos["vid"] is not None:
                fps, skip = 20, 2
                for key, infos in [("train", train_log_infos), ("test", test_log_infos)]:
                    frames = np.transpose(infos["vid"], (0, 3, 1, 2))
                    if infos["vid"].shape[0] > 1:
                        logger.log({f"media/{key}_video": wandb.Video(frames[::skip, :, :, :], fps=fps, format="gif")})
                    else:
                        logger.log({f"media/{key}_image": wandb.Video(frames[::skip, :, :, :], fps=fps, format="gif")})
                    logger.log(
                        {
                            f"media/{key}_step": step,
                            f"media/{key}_epoch": epoch,
                            f"media/{key}_episode_len": infos["episode_len"],
                        }
                    )

            if step == total_steps - 1:
                # Record few rollouts for valuation.
                for key, videos in [("train", train_videos), ("test", test_videos)]:
                    for video in videos:
                        frames = np.transpose(video, (0, 3, 1, 2))
                        fps, skip = 20, 2
                        if video.shape[0] > 1:
                            logger.log(
                                {
                                    f"media/{key}_final_video": wandb.Video(
                                        frames[::skip, :, :, :], fps=fps, format="gif"
                                    )
                                }
                            )
                        else:
                            logger.log(
                                {
                                    f"media/{key}_final_image": wandb.Video(
                                        frames[::skip, :, :, :], fps=fps, format="gif"
                                    )
                                }
                            )
            tqdm.write("\n" + pprint.pformat(train_log_metrics) + "\n")
            tqdm.write("\n" + pprint.pformat(test_log_metrics) + "\n")

        if step and step % save_model_freq == 0 or step == total_steps - 1:
            save_data = {
                "step": step,
                "epoch": epoch,
                "variant": variant,
                "state": jax.device_get(jax_utils.unreplicate(state)),
            }
            if jax_process_index == 0:
                logger.save_pickle(save_data, f"model_epoch{epoch}.pkl")
                if log_metrics.get("test_return", 0.0) > best_eval_score:
                    best_eval_score = log_metrics["test_return"]
                    logger.save_pickle(save_data, f"model_best.pkl")
                    logger.wandb_save_model("model_best.pkl")
                if step == total_steps - 1:
                    logger.wandb_save_model(f"model_epoch{epoch}.pkl")
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == "__main__":
    jax.config.config_with_absl()
    tf.config.experimental.set_visible_devices([], "GPU")
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
