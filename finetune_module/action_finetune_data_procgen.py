import clip
import gcsfs
import h5py
import numpy as np
import torch
from ml_collections import ConfigDict

from arp_dt.data_procgen import get_clip_instruct


class ProcgenActionDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = "../demonstrations"

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_size = 512
        config.num_frames = 8

        config.state_key = ""
        config.state_dim = 0

        config.image_key = "ob"
        config.augmentations = ""

        config.action_key = ""
        config.action_dim = 15

        config.num_demonstrations = 200
        config.window_size = 8

        # dealing with AISC type
        config.env_type = "none"

        config.k = 15
        config.target_ratio = 0.8
        config.threshold = 20

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, update, dataset_name="reach_target", start_offset_ratio=None, split="train"):
        self.config = self.get_default_config(update)
        assert self.config.path != ""

        self.dataset_name = dataset_name

        if split == "train":
            path = f"{self.config.path}/{dataset_name}/data_train.hdf5"
        elif split == "val":
            path = f"{self.config.path}/{dataset_name}/data_val.hdf5"

        if self.config.path.startswith("gs://"):
            self.h5_file = h5py.File(gcsfs.GCSFileSystem().open(path, cache_type="block"), "r")
        else:
            self.h5_file = h5py.File(path, "r")

        self.env_name = self.h5_file.attrs["env_name"]
        if self.config.env_type != "none":
            self.env_name = f"{self.env_name}_{self.config.env_type}"
        if "mugen" in self.config.path:
            self.env_name += "_mugen"

        if self.config.random_start:
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

        self.h5_file_traj_idx = self.get_traj_idx()
        self.idx_to_traj = self.get_idx_to_traj()

    def __getstate__(self):
        return self.config, self.random_start_offset, self.dataset_name

    def __setstate__(self, state):
        config, random_start_offset, dataset_name = state
        self.__init__(config, dataset_name=dataset_name)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["ob"].shape[0] - self.config.start_index,
            self.config.max_length,
        )

    def get_traj_idx(self):
        h5_file_traj_idx = list(np.nonzero(self.h5_file["done"][:, -1])[0] + 1)
        h5_file_traj_idx.insert(0, 0)
        return h5_file_traj_idx

    def get_idx_to_traj(self):
        idx_to_traj = {idx: 0 for idx in range(self.h5_file["done"].shape[0])}
        for traj_idx in range(len(self.h5_file_traj_idx) - 1):
            traj_indices = list(range(self.h5_file_traj_idx[traj_idx], self.h5_file_traj_idx[traj_idx + 1]))
            for ind in traj_indices:
                idx_to_traj[ind] = traj_idx

        return idx_to_traj

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    # def get_video_indices(self, traj_elems, idx):
    #     if idx + self.config.clip_frames <= traj_elems[-1]:
    #         return list(range(idx, idx + self.config.clip_frames))
    #     else:
    #         quotient = list(range(idx, traj_elems[-1] + 1))
    #         # remainder = [idx] * (self.config.clip_frames - len(quotient))
    # return quotient

    def sample_next_index(self, index, traj_elems):
        next_index = None
        threshold = min(int(len(traj_elems) * self.config.target_ratio), self.config.threshold)
        # print(f"threshold: {threshold}")
        trial, max_trial = 0, 10
        while trial < max_trial:
            next_index = np.random.choice(traj_elems, 2)
            if all([abs(ni - index) >= threshold for ni in next_index]):
                break
            trial += 1

        if trial >= max_trial:
            next_index = [max(index - threshold, traj_elems[0]), min(index + threshold, traj_elems[-1])]
        return next_index

    def __getitem__(self, index):
        # find the trajectory number of given index.
        index = self.process_index(index)
        traj_idx = self.idx_to_traj[index]
        traj_elems = list(range(self.h5_file_traj_idx[traj_idx], self.h5_file_traj_idx[traj_idx + 1]))

        # randomly sample k from future timesteps.
        # k = np.random.choice(range(1, self.config.k + 1), 1)[0]
        # next_index_1, next_index_2 = self.sample_next_index(index, traj_elems)
        # indices = sorted([index, next_index_1, next_index_2])
        indices = sorted([traj_elems[0], index, min(index + 1, traj_elems[-1]), traj_elems[-1]])
        # indices = sorted([index, index, index])

        res = {
            "image0": {},
            "image1": {},
            "image2": {},
            "image3": {},
            # "timestep1": None, "timestep2": None, "timestep3": None
        }
        for i, idx in enumerate(indices, start=0):
            # video_indices = self.get_video_indices(traj_elems, idx)
            for key in self.config.image_key.split(", "):
                obs_frames = self.h5_file[key][idx][-1]
                res[f"image{i}"][key] = obs_frames
                # res[f"timestep{i}"] = idx

        instruct = get_clip_instruct(self.env_name)
        res["r"] = np.array([int(indices[-2] == indices[-1])])
        res["instruct"] = clip.tokenize(instruct)
        res["action"] = self.h5_file["act"][indices[0]][-1]
        return res

    @property
    def num_actions(self):
        return self.config.action_dim


if __name__ == "__main__":
    config = ProcgenActionDataset.get_default_config()
    config.path = "/home/changyeon/procgen_generalization/procgen_data/clip/new_demonstrations_clip_ver3"
    dataset = ProcgenActionDataset(
        update=config,
        dataset_name="coinrun_hard_level0to500_num500_frame8",
    )

    random_indices = np.random.choice(range(len(dataset)), 10)
    for idx in random_indices:
        print(f"idx {idx}")
        elem = dataset[idx]
        for key, val in elem.items():
            if isinstance(val, np.ndarray):
                print(f"{key}: {val.shape}")
            elif isinstance(val, dict):
                for _key, _val in val.items():
                    print(f"{key}_{_key}: {_val.shape}")
            else:
                print(f"{key}: {val}")
        print()
