import copy
import logging
import os
from collections import defaultdict, deque

import clip
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm.auto import trange

from ..label_reward import center_crop
from .vl_reward import (
    device,
    get_torch_clip_adapter_goal_conditioned_reward,
    get_torch_clip_adapter_reward,
    get_torch_clip_goal_conditioned_reward,
    get_torch_clip_reward,
)


def batch_rollout(
    rng,
    data_aug_rng,
    env,
    policy_fn,
    transform_obs_fn,
    transform_action_fn,
    episode_length=2500,
    log_interval=None,
    window_size=4,
    num_episodes=1,
    return_to_go=100.0,
    scale=100.0,
    clip_model=None,
    vl_type="clip",
    text=None,
    reward_min=0.0,
    use_normalize=False,
    use_crop=False,
    eval_data_path=None,
    data_name="data.hdf5",
):
    concat_fn = lambda x, y: jnp.concatenate([x, y], axis=1)
    trim_fn = lambda x: x[:, -window_size:, ...]
    batch_fn = lambda x: x[None, None, ...]

    # get indices of test trajectories.
    if eval_data_path is not None:
        eval_hdf5 = h5py.File(os.path.join(eval_data_path, data_name), "r")
        eval_traj_idx = list(np.nonzero(eval_hdf5["done"][:, -1])[0] + 1)
        eval_traj_idx.insert(0, 0)
        assert len(eval_traj_idx) >= num_episodes

    def prepare_input(all_inputs, obs, rtg):
        action = np.zeros(1, dtype=np.int32)
        inputs = {**obs, "action": action, "rtg": rtg}
        inputs = jax.tree_util.tree_map(batch_fn, inputs)
        inputs["action"] = inputs["action"].squeeze(-1)

        if len(all_inputs) == 0:
            inputs = inputs
        else:
            all_inputs_copy = copy.deepcopy(all_inputs)
            inputs = jax.tree_util.tree_map(concat_fn, all_inputs_copy, inputs)
            inputs = jax.tree_util.tree_map(trim_fn, inputs)

        return all_inputs, inputs

    def update_input(all_inputs, obs, action, rtg):
        inputs = {**obs, "action": action, "rtg": rtg}
        inputs = jax.tree_util.tree_map(batch_fn, inputs)

        if len(all_inputs) == 0:
            all_inputs = inputs
        else:
            all_inputs = jax.tree_util.tree_map(concat_fn, all_inputs, inputs)
            all_inputs = jax.tree_util.tree_map(trim_fn, all_inputs)

        return all_inputs

    reward = jnp.zeros(1, dtype=jnp.float32)
    ep_lens = jnp.zeros(1, dtype=jnp.float32)

    videos = []
    for ep in trange(num_episodes, desc="rollout", ncols=0):
        episode_data = []
        rtg = {key: jnp.full(1, return_to_go / scale, dtype=jnp.float32) for key in env.config.image_key.split(", ")}
        all_inputs = {}
        done = jnp.zeros(1, dtype=jnp.int32)
        if eval_data_path is not None:
            goal_image = eval_hdf5["ob"][eval_traj_idx[ep + 1] - 1, -1]

        for t in trange(episode_length, desc=f"episode {ep}", ncols=0, leave=False):
            done_prev = done
            if t == 0:
                if eval_data_path is not None:
                    # reset environment using saved states
                    traj_state = np.load(os.path.join(eval_data_path, f"traj_state_{ep}.npy"), allow_pickle=True)
                    obs = env.reset()
                    env._env.env.env.env.set_state(traj_state[0])
                    rgb = env._env.env.env.env.observe()[1]["rgb"][0]
                    obs = env.get_image_state(rgb)
                    env._recorded_images[0] = rgb
                    # add goal_image in obs
                    obs["goal"] = {"ob": goal_image}
                else:
                    obs = env.reset(env.config.rand_seed + ep)
            else:
                obs = next_obs
            if transform_obs_fn is not None:
                input_obs = copy.deepcopy(obs)
                for key, val in input_obs["image"].items():
                    input_obs["image"][key], data_aug_rng = transform_obs_fn(val, data_aug_rng)
                if eval_data_path is not None:
                    for key, val in input_obs["goal"].items():
                        input_obs["goal"][key], data_aug_rng = transform_obs_fn(val, data_aug_rng)
            else:
                input_obs = obs

            all_inputs, inputs = prepare_input(all_inputs, input_obs, rtg)
            action = jax.device_get(policy_fn(inputs=inputs, rngs=rng))[0]
            action = transform_action_fn(action)
            all_inputs = update_input(all_inputs, input_obs, action, rtg)

            next_obs, _reward, done, info = env.step(action)
            if eval_data_path is not None:
                next_obs["goal"] = {"ob": goal_image}

            reward = reward + _reward * (1 - done_prev)
            if clip_model[0] is not None:
                for key in obs["image"].keys():
                    if vl_type == "clip":
                        clip_reward = get_torch_clip_reward(clip_model, obs["image"][key], text, use_crop=use_crop)
                    elif vl_type == "clip_goal_conditioned":
                        clip_reward = get_torch_clip_goal_conditioned_reward(
                            clip_model, obs["image"][key], goal_image, use_crop=use_crop
                        )
                    elif vl_type == "clip_ft_goal_conditioned":
                        clip_reward = get_torch_clip_adapter_goal_conditioned_reward(
                            clip_model, obs["image"][key], goal_image, use_crop=use_crop
                        )
                    elif vl_type == "clip_ft":
                        clip_reward = get_torch_clip_adapter_reward(
                            clip_model, obs["image"][key], text, use_crop=use_crop
                        )
                    else:
                        raise ValueError

                    if use_normalize:
                        rtg[key] -= ((clip_reward - reward_min[key])) / scale
                    else:
                        rtg[key] -= clip_reward / scale
            episode_data.append(
                {
                    "image": obs["image"],
                    "action": action,
                    "reward": _reward,
                    "state":  env._env.env.env.env.get_state(),
                    "done": done,
                    "info": info
                }
            )
            done = jnp.logical_or(done, done_prev).astype(jnp.int32)
            if log_interval and t % log_interval == 0:
                logging.info("step: %d done: %s reward: %s", t, done, reward)

            if jnp.all(done):
                ep_lens += info["episode_len"]
                np.save(os.path.join("/home/changyeon/procgen_"))
                break

        if info["vid"] is not None:
            videos.append(info["vid"])

    metric = {
        "return": reward.astype(jnp.float32) / num_episodes,
        "episode_length": ep_lens.astype(jnp.float32) / num_episodes,
    }
    return metric, info, videos
