import os
import pickle
from collections import defaultdict, deque
from typing import Any, Callable

import h5py
import numpy as np
from gym3.env import Env
from gym3.types import multimap
from gym3.types_np import concat, zeros
from gym3.wrapper import Wrapper


class TrajectoryRecorderWrapper(Wrapper):
    """
    Record a trajectory of each episode from an environment.

    Each saved file contains a single trajectory in pickle format, represented by a dictionary of lists of the same length as
    the trajectory. The dictionary keys are as follows:
    - "ob": list of observations
    - "act": list of actions. act[i] is the action taken after seeing ob[i]
    - "reward": list of rewards. reward[i] is the reward caused by taking act[i]
    - "info": list of metadata not observed by the agent. info[i] corresponds to the same timestep as ob[i]

    You can load a trajectory file like so:

        import pickle
        with open(filename, "rb") as f:
            trajectory = pickle.load(f)

    :param env: gym3 environment to record
    :param directory: directory to save trajectories to
    :param filename_prefix: use this prefix for the filenames of trajectories that are saved
    """

    def __init__(
        self,
        env: Env,
        env_fn: Callable,
        env_name: str,
        env_type: str,
        directory: str,
        data_type: str = "train",
        filename_prefix: str = "",
        save_trajectory: bool = True,
        save_type: str = "h5py",
        enable_filter: bool = True,
        num_frames: int = 8,
        use_random_action: bool = False,
        rand_seed: int = 42,
    ) -> None:
        super().__init__(env=env)
        self._env_fn = env_fn
        self._env_name = env_name
        self._env_type = env_type
        self._data_type = data_type
        self._num_frames = num_frames
        self._save_type = save_type
        self._enable_filter = enable_filter
        self._save_trajectory = save_trajectory
        self._use_random_action = use_random_action
        self._rand_seed = rand_seed

        if save_trajectory and self._save_type == "hdf5":
            self._hdf5_file = h5py.File(os.path.join(directory, f"data_{self._data_type}.hdf5"), "w")
            self._hdf5_file.attrs["env_name"] = self._env_name

        self._prefix = filename_prefix
        self._directory = os.path.abspath(directory)
        os.makedirs(self._directory, exist_ok=True)
        self._episode_count = 0
        self._trajectories = None
        self._ob_actual_dtype = None
        self._ac_actual_dtype = None
        self._traj_returns = []
        self._traj_lengths = []
        self._trial = 0

    def _new_trajectory_dict(self):
        assert self._ob_actual_dtype is not None, (
            "Not supposed to happen; self._ob_actual_dtype should have been set"
            " in the first act() call before _new_trajectory_dict is called"
        )
        traj_dict = dict(
            reward=list(),
            ob=zeros(self.env.ob_space, (0,)),
            info=list(),
            act=zeros(self.env.ac_space, (0,)),
            state=list(),
        )
        traj_dict["ob"] = multimap(
            lambda arr, my_dtype: arr.astype(my_dtype),
            traj_dict["ob"],
            self._ob_actual_dtype,
        )
        traj_dict["act"] = multimap(
            lambda arr, my_dtype: arr.astype(my_dtype),
            traj_dict["act"],
            self._ac_actual_dtype,
        )
        return traj_dict

    def stack_frames(self, data):
        total_data = defaultdict(list)
        stack = defaultdict(lambda: deque([], maxlen=self._num_frames))
        for k, v in data.items():
            for idx, elem in enumerate(v):
                if idx == 0:
                    stack[k].extend([elem] * self._num_frames)
                else:
                    stack[k].append(elem)
                total_data[k].append(np.stack(stack[k]))

        data = {k: np.asarray(v) for k, v in total_data.items()}
        return data

    def _check_to_write(self, idx):
        intermediate = (
            (
                filter_condition(
                    self._env_name, self._env_type, self._trajectories[idx], use_random_action=self._use_random_action
                )
            )
            if self._enable_filter
            else True
        )
        return intermediate and len(self._trajectories[idx]["ob"]) < 1000

    def _create_env(self):
        new_env = self._env_fn(rand_seed=self._rand_seed + self._trial)
        del self.env
        self.env = new_env

    def _write_and_reset_trajectory(self, idx) -> None:
        self._trial += 1
        write = self._check_to_write(idx)
        if write:
            data = self._trajectories[idx]
            data["reward"] = np.array(data["reward"])

            data["done"] = np.zeros_like(data["reward"])
            data["done"][-1] = 1.0

            data["success"] = np.zeros_like(data["reward"])
            data["success"][-1] = 1.0 if np.sum(data["reward"]) >= 10.0 else 0.0

            if self._save_trajectory:
                target_keys = ["ob", "act", "done", "reward", "success"]
                if self._save_type == "hdf5":
                    # convert to h5py
                    data = self.stack_frames(data)
                    if self._episode_count == 0:
                        for key in target_keys:
                            if key == "ob":
                                v_shape = data[key].shape[-3:]
                                self._hdf5_file.create_dataset(
                                    key,
                                    compression="gzip",
                                    data=data[key],
                                    chunks=(1, self._num_frames, *v_shape),
                                    maxshape=(None, self._num_frames, *v_shape),
                                )
                            else:
                                self._hdf5_file.create_dataset(
                                    key,
                                    compression="gzip",
                                    data=data[key],
                                    chunks=(1, self._num_frames),
                                    maxshape=(None, self._num_frames),
                                )
                    else:
                        for key in data:
                            if key in target_keys:
                                _dataset, _data = self._hdf5_file[key], data[key]
                                _dataset.resize((_dataset.shape[0] + _data.shape[0]), axis=0)
                                _dataset[-_data.shape[0] :] = _data

                elif self._save_type == "npy":
                    filepath = os.path.join(self._directory, f"{self._prefix}{self._episode_count:03d}.npy")
                    np.save(filepath, data, allow_pickle=True)

                elif self._save_type == "pickle":
                    filepath = os.path.join(self._directory, f"{self._prefix}{self._episode_count:03d}.pickle")
                    with open(filepath, "wb") as f:
                        pickle.dump(data, f)

            self._traj_returns.append(np.sum(data["reward"]))
            self._traj_lengths.append(len(data["reward"]))
            self._episode_count += 1
        # reset data.
        self._trajectories[idx] = self._new_trajectory_dict()
        return write

    def act(self, ac: Any) -> bool:
        ep_finished, writed = False, False
        _, ob, _ = self.observe()
        info = self.get_info()

        # We have to wait for the first call to act() to initialize the _trajectories list, because
        # sometimes the environment returns observations with dtypes that do not match self.env.ob_space.
        if self._trajectories is None:
            self._ob_actual_dtype = multimap(lambda x: x.dtype, ob)
            self._ac_actual_dtype = multimap(lambda x: x.dtype, ac)
            self._trajectories = [self._new_trajectory_dict() for _ in range(self.env.num)]

        for i in range(self.env.num):
            # With non-dict spaces, the `ob` and/or `ac` is a numpy array of shape [batch, obs_shape...] so separating
            # each trajectory into its own structure was relatively simple.
            # Take ob[i] then append it to self._trajectories[i]['ob'].
            #
            # With dict spaces, the returned ob becomes a nested dict
            # {
            #     'obs_key1': [batch, obs1_shape...],
            #     'obs_key2': [batch, obs2_shape...]
            # }
            # So to separate each trajectory, we have to take ob['obs_key1'][i] then append it to
            # self._trajectories[i]['ob']['obs_key1']
            self._trajectories[i]["ob"] = concat(
                [self._trajectories[i]["ob"], multimap(lambda x: x[i : i + 1], ob)],
                axis=0,
            )
            self._trajectories[i]["act"] = concat(
                [self._trajectories[i]["act"], multimap(lambda x: x[i : i + 1], ac)],
                axis=0,
            )
            self._trajectories[i]["info"].append(info[i])

        super().act(ac)

        reward, _, first = self.observe()
        for i in range(self.env.num):
            self._trajectories[i]["reward"].append(reward[i])

        # For each completed trajectory, write it out
        for i in range(self.env.num):
            if first[i]:
                writed = self._write_and_reset_trajectory(i)
                self._create_env()
                ep_finished = True

        return ep_finished, writed


def filter_condition(env_name, env_type, data, use_random_action=False):
    sum_rewards = np.sum(data["reward"])
    if env_name in ["bossfight", "maze", "caveflyer"]:
        return sum_rewards >= 10.0
    elif env_name in ["coinrun"]:
        return sum_rewards == 0.0 if use_random_action else sum_rewards >= 10.0
    elif env_name in ["starpilot"]:
        return sum_rewards >= 30.0
    elif env_name in ["heist"]:
        if env_type == "none":
            return sum_rewards >= 10.0
        elif "aisc" in env_type:
            return sum_rewards > 0.0
    elif env_name in ["bigfish"]:
        return sum_rewards >= 1.0
    else:
        raise NotImplementedError
