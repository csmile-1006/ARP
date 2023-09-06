from typing import Union

import gym
import numpy as np
from ml_collections import ConfigDict


class Procgen:
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.image_key = "ob"
        config.state_key = ""
        config.episode_length = 1000
        config.record_video = True
        config.record_every = 50

        config.distribution_mode = "hard"
        config.num_levels = 500
        config.start_level = 0
        config.eval_start_level = 500
        config.rand_seed = 42

        config.eval_env_type = "none"
        config.use_train_levels = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, game_name: str, update: ConfigDict, image_resolution: str = "high"):
        self.config = self.get_default_config(update)

        self._episode_index = 0
        self._record_current_episode = True
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._i = 0
        self._image_resolution = image_resolution

        self.game_name = game_name
        self._create_env()

    def _create_env(self, rand_seed=42):
        # change evaluation level for more harsh setting.
        if self.config.use_train_levels:
            num_levels = self.config.num_levels
            start_level = self.config.start_level
        else:
            num_levels = self.config.num_levels * 2
            start_level = self.config.start_level + self.config.num_levels

        kwargs = dict(
            distribution_mode=self.config.distribution_mode,
            num_levels=num_levels,
            start_level=start_level,
            rand_seed=rand_seed,
        )
        if self.config.eval_env_type == "none":
            if self._image_resolution == "high":
                env = gym.make(id=f"procgen-highres-{self.game_name}-v0", **kwargs)
            elif self._image_resolution == "low":
                env = gym.make(id=f"procgen-{self.game_name}-v0", **kwargs)
        elif self.config.eval_env_type != "none":
            if self._image_resolution == "high":
                env = gym.make(id=f"procgen-highres-aisc-{self.game_name}_{self.config.eval_env_type}-v0", **kwargs)
            elif self._image_resolution == "low":
                env = gym.make(id=f"procgen-aisc-{self.game_name}-v0", **kwargs)
        self._env = env

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, rand_seed=42):
        self._create_env(rand_seed=rand_seed)
        obs = self._env.reset()
        self._prev_obs = obs
        res = self.get_image_state(obs)

        self._i = 0
        self._episode_index += 1
        self._record_current_episode = self.config.record_video and self._episode_index % self.config.record_every == 0

        self._recorded_images.clear()
        self.record(obs)
        return res

    def record(self, obs):
        self._recorded_images.append(obs)

    def step(self, action: Union[int, np.ndarray]):
        obs, reward, terminal, _ = self._env.step(action)
        self.record(obs)
        self._prev_obs = obs

        res = self.get_image_state(obs)

        self._i += 1

        if terminal or self._i == self.config.episode_length:
            done = True
            if self._record_current_episode:
                vid = np.array(self._recorded_images)
            else:
                vid = None
        else:
            done = False
            vid = None

        info = {"vid": vid, "episode_len": self._i, "terminal": terminal}

        return res, reward, done, info

    def get_image_state(self, obs):
        res = {"image": {}}
        for k in self.config.image_key.split(", "):
            res["image"][k] = obs
        if self.config.state_key != "":
            res["state"] = np.concatenate([obs[k] for k in self.config.state_key.split(", ")])
        return res


if __name__ == "__main__":
    config = Procgen.get_default_config()
    config.record_video = True
    config.episode_length = 10
    config.record_every = 1
    env = Procgen("coinrun", config)
    init = env.reset()
    timestep = 0
    for _ in range(100):
        timestep += 1
        res, rew, done, info = env.step(env.action_space.sample())
        print(timestep)
        if done:
            break
