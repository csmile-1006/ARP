import argparse

import numpy as np
import phasic_policy_gradient.torch_util as tu
import torch
from phasic_policy_gradient.envs import get_venv
from tqdm import tqdm
from trajectory_recorder import TrajectoryRecorderWrapper


def evaluation_ppg(
    model_path, num_envs, env_name, distribution_mode, num_levels, start_level, env_type="none", num_eval_episodes=200
):
    seed = 42
    trained_model = torch.load(model_path)
    lowres_venv = get_venv(
        num_envs=num_envs,
        env_name=env_name,
        env_type=env_type,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=start_level,
        rand_seed=seed,
    )
    lowres_venv = TrajectoryRecorderWrapper(
        env=lowres_venv, env_name=env_name, env_type=env_type, directory="", save_trajectory=False, enable_filter=False
    )

    _state = trained_model.initial_state(lowres_venv.num)
    num_episodes = 0
    traj_returns = []
    traj_lengths = []
    with tqdm(total=num_eval_episodes) as pbar:
        while True:
            _, ob, first = lowres_venv.observe()
            ac, newstate, _ = trained_model.act(ob=tu.np2th(ob), first=first, state_in=_state)

            _state = newstate
            ep_finished, _ = lowres_venv.act(tu.th2np(ac))

            if ep_finished:
                seed += 1
                num_episodes += 1
                traj_returns.extend(lowres_venv._traj_returns)
                traj_lengths.extend(lowres_venv._traj_lengths)
                lowres_venv = get_venv(
                    num_envs=num_envs,
                    env_name=env_name,
                    env_type=env_type,
                    distribution_mode=distribution_mode,
                    num_levels=num_levels,
                    start_level=start_level,
                    rand_seed=seed,
                )
                lowres_venv = TrajectoryRecorderWrapper(
                    env=lowres_venv,
                    env_name=env_name,
                    env_type=env_type,
                    directory="",
                    save_trajectory=False,
                    enable_filter=False,
                )
                pbar.update(1)

            if num_episodes >= num_eval_episodes:
                break

    avg_return = np.mean(traj_returns)
    avg_length = np.mean(traj_lengths)
    print(f"Average Return of {num_episodes} episodes: {avg_return:.3f}")
    print(f"Average Length of {num_episodes} episodes: {avg_length:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Process rollout training arguments.")
    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--env_type", type=str, default="none")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_levels", type=int, default=200)
    parser.add_argument("--start_level", type=int, default=200)
    parser.add_argument("--distribution_mode", type=str, default="easy")
    parser.add_argument("--model_path", type=str, required=True, default=None)
    parser.add_argument("--num_eval_episodes", type=int, default=200)

    args = parser.parse_args()

    evaluation_ppg(
        model_path=args.model_path,
        num_envs=args.num_envs,
        env_name=args.env_name,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        env_type=args.env_type,
        num_eval_episodes=args.num_eval_episodes,
    )


if __name__ == "__main__":
    main()
