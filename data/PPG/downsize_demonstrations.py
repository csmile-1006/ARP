import argparse
import os
from collections import deque

import h5py
import numpy as np
from phasic_policy_gradient.envs import get_venv
from tqdm import tqdm


def stack_frames(data, num_frames):
    total_data = []
    stack = deque([], maxlen=num_frames)
    for idx, elem in enumerate(data):
        if idx == 0:
            stack.extend([elem] * num_frames)
        else:
            stack.append(elem)
        total_data.append(np.stack(stack))
    return total_data


def downsize_demonstrations(origin_path, num_envs, env_name, distribution_mode, num_levels, start_level, num_frames):
    lowres_venv = get_venv(
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=start_level,
    )
    origin_dataset = h5py.File(os.path.join(origin_path, "data.hdf5"), "r")
    downsize_dataset = h5py.File(os.path.join(origin_path, "data_64x64.hdf5"), "w")

    downsize_dataset.attrs["env_name"] = origin_dataset.attrs["env_name"]
    for obj in origin_dataset.keys():
        if obj != "ob":
            origin_dataset.copy(obj, downsize_dataset)
    origin_dataset.close()

    downsize_dataset.create_dataset(
        "ob",
        compression="gzip",
        chunks=True,
        shape=(downsize_dataset["act"].shape[0], num_frames, 64, 64, 3),
        dtype=np.uint8,
    )
    env_state_files = [file for file in os.listdir(origin_path) if file.endswith(".npy")]
    env_state_files = sorted(env_state_files, key=lambda x: int(x.split("_")[-1][:-4]))

    div = 2
    cursor = 0
    for file in tqdm(env_state_files, ncols=0, desc="render low res obs."):
        states = np.load(os.path.join(origin_path, file), allow_pickle=True)
        num = len(states)
        num_envs = [num // div + (1 if x < num % div else 0) for x in range(div)]
        obs = []

        n = 0
        for _num_env in num_envs:
            lowres_venv = get_venv(
                num_envs=_num_env,
                env_name=env_name,
                distribution_mode=distribution_mode,
                num_levels=num_levels,
                start_level=start_level,
            )
            lowres_venv.callmethod("set_state", states[n : n + _num_env].squeeze(-1))
            _, low_ob, _ = lowres_venv.observe()
            obs.extend(low_ob)
            # downsize_dataset['ob'][cursor:cursor + _num_env] =
            n += _num_env

        stacked_obs = stack_frames(obs, num_frames)
        downsize_dataset["ob"][cursor : cursor + num] = stacked_obs
        cursor += num
        # for state in tqdm(states, leave=False):
        #     lowres_venv.callmethod("set_state", state)
        #     _, low_ob, _ = lowres_venv.observe()
        #     downsize_dataset['ob'][cursor] = low_ob
        #     cursor += 1

    assert cursor == downsize_dataset["act"].shape[0], f"{cursor} != {downsize_dataset['ac'].shape[0]}"

    downsize_dataset.close()


def main():
    parser = argparse.ArgumentParser(description="Process for changing resolutions for rendering")

    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_levels", type=int, default=200)
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--distribution_mode", type=str, default="easy")
    parser.add_argument("--num_demonstrations", type=int, default=100)

    parser.add_argument("--origin_path", type=str, required=True, default=None)
    parser.add_argument("--num_frames", type=int, default=8)

    args = parser.parse_args()

    assert os.path.exists(args.origin_path)
    filename = os.path.join(
        args.origin_path,
        f"{args.env_name}_{args.distribution_mode}_level{args.start_level}to{args.num_levels}_num{args.num_demonstrations}",
    )

    downsize_demonstrations(
        origin_path=filename,
        num_envs=args.num_envs,
        env_name=args.env_name,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
