import argparse
import os
from functools import partial

import numpy as np
import torch
from phasic_policy_gradient import torch_util as tu
from phasic_policy_gradient.envs import get_venv
from tqdm import tqdm
from trajectory_recorder import TrajectoryRecorderWrapper


def collect_demonstrations(
    model_dir,
    num_envs,
    env_name,
    distribution_mode,
    num_levels,
    start_level,
    env_type="none",
    data_type="train",
    num_demonstrations=100,
    save_dir=None,
    save_type="npy",
    num_frames=8,
    enable_filter=True,
    use_random_action=False,
    rand_seed=42,
):
    os.makedirs(save_dir, exist_ok=True)
    trained_model = torch.load(model_dir)

    lowres_venv = get_venv(
        num_envs=num_envs,
        env_name=env_name,
        env_type=env_type,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=start_level,
        high_res=False,
        rand_seed=rand_seed,
    )
    highres_venv = get_venv(
        num_envs=num_envs,
        env_name=env_name,
        env_type=env_type,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=start_level,
        high_res=True,
        rand_seed=rand_seed,
    )
    highres_venv_fn = partial(
        get_venv,
        num_envs=num_envs,
        env_name=env_name,
        env_type=env_type,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=start_level,
        high_res=True,
    )
    highres_venv = TrajectoryRecorderWrapper(
        env=highres_venv,
        env_fn=highres_venv_fn,
        env_name=env_name,
        env_type=env_type,
        data_type=data_type,
        directory=save_dir,
        filename_prefix="traj",
        save_type=save_type,
        num_frames=num_frames,
        enable_filter=enable_filter,
        use_random_action=use_random_action,
        rand_seed=rand_seed,
    )
    assert highres_venv.num == 1, "use single environment for compatibility."

    num_episodes = 0
    _state = trained_model.initial_state(highres_venv.num)
    env_states = []

    with tqdm(total=num_demonstrations) as pbar:
        while True:
            _, _, first = highres_venv.observe()
            states = highres_venv.callmethod("get_state")
            env_states.append(states)
            lowres_venv.callmethod("set_state", states)
            _, low_ob, _ = lowres_venv.observe()

            if use_random_action:
                ratio = np.random.rand()
                if ratio > 0.5:
                    ac = np.array([np.random.randint(15)])
                    ep_finished, writed = highres_venv.act(ac)
                else:
                    ac, newstate, _ = trained_model.act(ob=tu.np2th(low_ob), first=first, state_in=_state)
                    _state = newstate
                    ep_finished, writed = highres_venv.act(tu.th2np(ac))
            else:
                ac, newstate, _ = trained_model.act(ob=tu.np2th(low_ob), first=first, state_in=_state)
                _state = newstate
                ep_finished, writed = highres_venv.act(tu.th2np(ac))

            if ep_finished:
                if writed:
                    np.save(
                        os.path.join(save_dir, f"{data_type}_traj_state_{num_episodes}.npy"),
                        env_states,
                        allow_pickle=True,
                    )
                    num_episodes += 1
                    pbar.update(1)
                env_states = []

            if num_episodes >= num_demonstrations:
                break


def main():
    parser = argparse.ArgumentParser(description="Process rollout training arguments.")
    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--env_type", type=str, default="none")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_levels", type=int, default=200)
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--distribution_mode", type=str, default="easy")
    parser.add_argument("--arch", type=str, default="dual")
    parser.add_argument("--data_type", choices=["train", "val", "test"], default="train")

    parser.add_argument("--model_dir", type=str, required=True, default=None)
    parser.add_argument("--output_dir", type=str, default="./demonstrations")
    parser.add_argument("--num_demonstrations", type=int, default=100)
    parser.add_argument("--save_type", type=str, default="hdf5", choices=["npy", "hdf5"])
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--filter", action="store_true")
    parser.set_defaults(filter=True)
    parser.add_argument("--no_filter", dest="filter", action="store_false")
    parser.add_argument("--use_random_action", type=bool, default=False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(
        args.output_dir,
        f"{args.env_name}_{args.distribution_mode}_level{args.start_level}to{args.num_levels}_num{args.num_demonstrations}_frame{args.num_frames}",
    )
    if args.env_type != "none":
        save_dir += f"_{args.env_type}"
    if not args.filter:
        save_dir += "_unfiltered"
    if args.use_random_action:
        save_dir += "_random_action"

    collect_demonstrations(
        model_dir=args.model_dir,
        num_envs=args.num_envs,
        env_name=args.env_name,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        env_type=args.env_type,
        data_type=args.data_type,
        num_demonstrations=args.num_demonstrations,
        save_dir=save_dir,
        save_type=args.save_type,
        num_frames=args.num_frames,
        enable_filter=args.filter,
        use_random_action=args.use_random_action,
    )


if __name__ == "__main__":
    main()
