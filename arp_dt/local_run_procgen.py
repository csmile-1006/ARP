import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import pprint
from functools import partial

import absl.app
import absl.flags
import augmax
import clip
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
import wandb
from absl import app, logging
from tqdm.auto import tqdm

from .ARPDT import ARPDT
from .BC import BC
from .data_procgen import ProcgenDataset, get_clip_instruct, get_m3ae_instruct
from .envs import rollout_procgen
from .envs.procgen import Procgen
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
    load_checkpoint="",
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
    data=ProcgenDataset.get_default_config(),
    model=ARPDT.get_default_config(),
    env=Procgen.get_default_config(),
    window_size=4,
    episode_length=500,
    instruct="",
    num_test_episodes=5,
    num_actions=15,
    game_name="coinrun",
    use_text=False,
    instruct_length="more_short",
    tokenizer_max_length=77,
    return_to_go=100.0,
    scale=100.0,
    use_vl=True,
    use_task_reward=False,
    vl_type="clip",
    use_normalize=False,
    vl_checkpoint="",
    eval_with_goal=False
)
FLAGS = absl.flags.FLAGS


def build_env_fn(game_name):
    def env_fn():
        env = Procgen(game_name, FLAGS.env)
        return env

    return env_fn


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


def create_test_step(
    model,
    env_fn,
    episode_length,
    instruct,
    window_size,
    num_episodes,
    transform_obs_fn,
    transform_action_fn,
    return_to_go,
    scale,
    clip_model,
    vl_type,
    text,
    reward_min,
    use_normalize,
    eval_data_path,
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
        metric, _, videos = rollout_procgen.batch_rollout(
            rng=rng_generator(model.rng_keys()),
            data_aug_rng=rng_generator(),
            env=env_fn,
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
            eval_data_path=eval_data_path,
            data_name="data_test.hdf5",
        )
        return metric, videos, next_rng

    return test_step_fn


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    jax_devices = jax.local_devices()

    FLAGS.logging.experiment_name = "-".join(
        [
            FLAGS.game_name,
            FLAGS.env.distribution_mode,
            f"{FLAGS.env.start_level + FLAGS.env.num_levels}to{FLAGS.env.start_level + FLAGS.env.num_levels * 2}",
            f"{'no_text' if not FLAGS.use_text else ''}",
            f"note@{'+'.join(FLAGS.logging.notes.split(' '))}",
        ]
    )
    FLAGS.logging.project = "EVAL_" + FLAGS.logging.project

    # First setting for use discrete action in Procgen Benchmark.
    FLAGS.model.use_discrete_action = True

    if not FLAGS.use_vl:
        # If not use clip, baseline would be InstructRL with text representation.
        FLAGS.use_text = True
        variant["use_text"] = True

    dataset_name = f"{FLAGS.game_name}_{FLAGS.env.distribution_mode}_level{FLAGS.env.start_level}to{FLAGS.env.num_levels}_num{FLAGS.data.num_demonstrations}_frame{FLAGS.data.num_frames}"
    if not FLAGS.data.enable_filter:
        dataset_name += "_unfiltered"
    if FLAGS.data.train_env_type != "none":
        dataset_name += f"_{FLAGS.data.train_env_type}"

    train_dataset = ProcgenDataset(FLAGS.data, dataset_name, jax_process_index / jax_process_count)
    if FLAGS.eval_with_goal:
        test_data_path = os.path.join(
            FLAGS.data.path,
            f"{FLAGS.game_name}_{FLAGS.env.distribution_mode}_level{FLAGS.env.start_level+FLAGS.env.num_levels}to{FLAGS.env.num_levels*2}_num{FLAGS.num_test_episodes * 10}_frame{FLAGS.data.num_frames}",
        )
        if FLAGS.env.eval_env_type != "none":
            test_data_path += f"_{FLAGS.env.eval_env_type}"
    else:
        test_data_path = None

    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
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

    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    if FLAGS.use_vl or FLAGS.data.use_task_reward:
        model = ARPDT(
            config_updates=FLAGS.model,
            num_actions=FLAGS.num_actions,
            patch_dim=16,
            normalize_quterion=False,
        )
    else:
        model = BC(config_updates=FLAGS.model, num_actions=FLAGS.num_actions, patch_dim=16, normalize_quterion=False)

    def tokenize_fn(text):
        if FLAGS.model.transfer_type.startswith("clip"):
            from .models.openai import tokenizer

            token_fn = tokenizer.build_tokenizer()
            tokenized_text = token_fn(text).astype(np.int32)
            padding_mask = np.ones(FLAGS.tokenizer_max_length, dtype=np.float32)

        elif FLAGS.model.transfer_type.startswith("m3ae"):
            import transformers

            tokenizer = partial(
                transformers.BertTokenizer.from_pretrained("bert-base-uncased"),
                padding="max_length",
                truncation=True,
                max_length=FLAGS.tokenizer_max_length,
                return_tensors="np",
                add_special_tokens=False,
            )
            encoded_instruct = tokenizer(text)
            tokenized_text = encoded_instruct["input_ids"].astype(np.int32)
            padding_mask = 1.0 - encoded_instruct["attention_mask"].astype(np.float32)
        else:
            assert False, f"{FLAGS.instruct} not supported with {FLAGS.model.transfer_type}"
        return tokenized_text, padding_mask

    test_instruct = {"instruct": None, "text_padding_mask": None}
    instruct = FLAGS.instruct if FLAGS.instruct != "" else get_m3ae_instruct(FLAGS.game_name)
    if FLAGS.use_text:
        instruct_token, instruct_padding_mask = tokenize_fn(instruct)
        test_instruct = {"instruct": instruct_token, "text_padding_mask": instruct_padding_mask}

    transform_action_fn = lambda x: x

    clip_model, preprocess, text = None, None, None
    if FLAGS.use_vl:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        game_name = (
            FLAGS.game_name if FLAGS.env.eval_env_type == "none" else f"{FLAGS.game_name}_{FLAGS.env.eval_env_type}"
        )
        text = get_clip_instruct(game_name)
        if FLAGS.vl_type == "clip":
            clip_model, preprocess = clip.load("ViT-B/16", device=device)
        elif FLAGS.vl_type.startswith("clip_"):
            _, preprocess = clip.load("ViT-B/16", device=device)
            if FLAGS.vl_type == "clip_ft":
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
            clip_model.load_state_dict(model_state_dict)
            clip_model.eval()
        assert clip_model is not None

    assert FLAGS.load_checkpoint != "", "load_checkpoint is required"
    checkpoint_data = load_pickle(FLAGS.load_checkpoint)
    state = checkpoint_data["state"]

    env_fn = build_env_fn(FLAGS.game_name)
    test_step_fn = create_test_step(
        model=model,
        env_fn=env_fn(),
        episode_length=FLAGS.episode_length,
        instruct=test_instruct,
        window_size=FLAGS.window_size,
        num_episodes=FLAGS.num_test_episodes,
        transform_obs_fn=test_image_aug,
        transform_action_fn=transform_action_fn,
        return_to_go=getattr(train_dataset, "return_to_go", 1000.0),
        scale=getattr(train_dataset, "scale", 100.0),
        clip_model=(clip_model, preprocess),
        vl_type=FLAGS.vl_type,
        text=text,
        reward_min=train_dataset.reward_min,
        use_normalize=FLAGS.use_normalize,
        eval_data_path=test_data_path,
    )

    log_metrics, videos, _ = test_step_fn(state, next_rng())
    log_metrics = {k: v for k, v in log_metrics.items() if k in ["return", "episode_length"]}
    log_metrics = {f"test_{k}": np.array(v) for k, v in jax.tree_map(lambda x: x.mean(), log_metrics).items()}
    logger.log(log_metrics)
    for video in videos:
        frames = np.transpose(video, (0, 3, 1, 2))
        fps, skip = 30, 2
        if video.shape[0] > 1:
            logger.log({"media/video": wandb.Video(frames[::skip, :, :, :], fps=fps, format="gif")})
        else:
            logger.log({"media/image": wandb.Video(frames[::skip, :, :, :], fps=fps, format="gif")})

    tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == "__main__":
    jax.config.config_with_absl()
    tf.config.experimental.set_visible_devices([], "GPU")
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
