import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm, trange


def fuse_data(
    env_name,
    distribution_mode,
    num_levels,
    start_level,
    src_env_type="none",
    dst_env_type="aisc",
    num_demonstrations=500,
    num_frames=8,
    fuse_ratio=0.9,
    base_path="",
    output_path=None,
):
    dirname = (
        f"{env_name}_{distribution_mode}_level{start_level}to{num_levels}_num{num_demonstrations}_frame{num_frames}"
    )
    src_env_tp = "" if src_env_type == "none" else f"_{src_env_type}"
    dst_env_tp = "" if dst_env_type == "none" else f"_{dst_env_type}"
    f = h5py.File(os.path.join(base_path, f"{dirname}{src_env_tp}", "data.hdf5"), "r")
    g = h5py.File(os.path.join(base_path, f"{dirname}{dst_env_tp}", "data.hdf5"), "r")

    f_traj_idx = np.nonzero(f["done"][:, -1])[0] + 1
    g_traj_idx = np.nonzero(g["done"][:, -1])[0] + 1

    os.makedirs(
        os.path.join(output_path, f"ratio{fuse_ratio}", f"src_{src_env_type}_dst_{dst_env_type}", dirname),
        exist_ok=True,
    )
    h = h5py.File(
        os.path.join(output_path, f"ratio{fuse_ratio}", f"src_{src_env_type}_dst_{dst_env_type}", dirname, "data.hdf5"),
        "w",
    )
    h.attrs["env_name"] = f.attrs["env_name"]

    original_num, aisc_num = int(len(f_traj_idx) * fuse_ratio), round(len(f_traj_idx) * (1 - fuse_ratio))

    original_indices = f_traj_idx[original_num - 1]
    asic_indices = g_traj_idx[aisc_num - 1]

    for key in tqdm(f.keys()):
        for i in trange(0, original_indices, 1, leave=False, desc=f"src data: ({src_env_type})"):
            data = f[key][i : min(i + 1, original_indices)]
            if i == 0:
                if key == "ob":
                    num_frames, v_shape = data.shape[-4], data.shape[-3:]
                    h.create_dataset(
                        key,
                        compression="gzip",
                        data=data[:original_indices],
                        chunks=(1, num_frames, *v_shape),
                        maxshape=(None, num_frames, *v_shape),
                    )
                else:
                    num_frames = data.shape[-1]
                    h.create_dataset(
                        key,
                        compression="gzip",
                        data=data[:original_indices],
                        chunks=(1, num_frames),
                        maxshape=(None, num_frames),
                    )
            else:
                h[key].resize((h[key].shape[0] + data.shape[0]), axis=0)
                h[key][-data.shape[0] :] = data
        for j in trange(0, asic_indices, 1, leave=False, desc=f"dst data: ({dst_env_type})"):
            _dataset, _data = h[key], g[key][j : min(j + 1, asic_indices)]
            _dataset.resize((_dataset.shape[0] + _data.shape[0]), axis=0)
            _dataset[-_data.shape[0] :] = _data


def main():
    parser = argparse.ArgumentParser(description="Process rollout training arguments.")
    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--src_env_type", type=str, default="none")
    parser.add_argument("--dst_env_type", type=str, default="none")
    parser.add_argument("--num_levels", type=int, default=500)
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--distribution_mode", type=str, default="hard")

    parser.add_argument("--base_path", type=str, default="./demonstrations")
    parser.add_argument("--output_path", type=str, default="./demonstrations")
    parser.add_argument("--num_demonstrations", type=int, default=500)
    parser.add_argument("--fuse_ratio", type=float, default=0.9)
    parser.add_argument("--save_type", type=str, default="npy", choices=["npy", "hdf5"])
    parser.add_argument("--num_frames", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    fuse_data(
        env_name=args.env_name,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        src_env_type=args.src_env_type,
        dst_env_type=args.dst_env_type,
        num_demonstrations=args.num_demonstrations,
        num_frames=args.num_frames,
        fuse_ratio=args.fuse_ratio,
        base_path=args.base_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
