from collections import deque

import gcsfs
import h5py
import numpy as np
import torch
from ml_collections import ConfigDict

from arp_dt.utils import compute_scale


class ProcgenDataset(torch.utils.data.Dataset):
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
        config.action_dim = 15

        config.num_demonstrations = 200
        config.num_subset = -1
        config.window_size = 8

        config.use_bert_tokenizer = True
        config.tokenizer_max_length = 77

        config.augmentations = "random_crop,color_jitter"

        # Filter only successful trajectories.
        config.enable_filter = True

        # DT-style only option
        config.scale = 100.0
        config.use_task_reward = False
        config.use_normalize = False

        # dealing with AISC type
        config.train_env_type = "none"
        config.use_vl = False
        config.vl_type = "clip"
        config.inst_type = "none"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, update, dataset_name="reach_target", start_offset_ratio=None, split="train"):
        self.config = self.get_default_config(update)
        assert self.config.path != ""

        self.dataset_name = dataset_name
        self.split = split

        if split == "train":
            path = f"{self.config.path}/{dataset_name}/data_train.hdf5"
        elif split == "val":
            path = f"{self.config.path}/{dataset_name}/data_val.hdf5"

        if self.config.path.startswith("gs://"):
            self.h5_file = h5py.File(gcsfs.GCSFileSystem().open(path, cache_type="block"), "r")
        else:
            self.h5_file = h5py.File(path, "r")

        # self.env_name = self.h5_file.attrs["env_name"]
        self.env_name = dataset_name.split("_")[0]
        if self.config.train_env_type != "none":
            self.env_name = f"{self.env_name}_{self.config.train_env_type}"

        # check the validity of frames
        h5_file_num_frames = self.h5_file["ob"][0].shape[1]
        assert (
            h5_file_num_frames > self.config.window_size
        ), f"this file have {h5_file_num_frames} stacked frames < window_size {self.config.window_size}"
        self.window_size = self.config.window_size

        if self.config.random_start:
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

        self.tokenizer = self.build_tokenizer()
        self.h5_file_traj_idx = self.get_traj_idx()
        self.idx_to_traj = self.index_to_traj()
        if self.config.use_vl:
            self.rtgs = self.preprocess_rtgs()

    def __getstate__(self):
        return self.config, self.random_start_offset, self.dataset_name

    def __setstate__(self, state):
        config, random_start_offset, dataset_name = state
        self.__init__(config, dataset_name=dataset_name)
        self.random_start_offset = random_start_offset

    def __len__(self):
        if self.split == "train" and self.config.num_subset != -1:
            h5_file_traj_idx = self.get_traj_idx()
            return h5_file_traj_idx[self.config.num_subset]
        else:
            return min(
                self.h5_file["ob"].shape[0] - self.config.start_index,
                self.config.max_length,
            )

    def get_traj_idx(self):
        h5_file_traj_idx = list(np.nonzero(self.h5_file["done"][:, -1])[0] + 1)
        h5_file_traj_idx.insert(0, 0)
        return h5_file_traj_idx

    def index_to_traj(self):
        h5_file_traj_idx = list(np.nonzero(self.h5_file["done"][:, -1])[0] + 1)
        h5_file_traj_idx.insert(0, 0)
        idx_to_traj = np.zeros_like(self.h5_file["done"][:, -1], dtype=np.int32)
        for idx in range(len(h5_file_traj_idx) - 1):
            traj_range = list(range(h5_file_traj_idx[idx], h5_file_traj_idx[idx + 1]))
            idx_to_traj[traj_range] = idx
        return idx_to_traj

    def preprocess_rtgs(self):
        def discount_cumsum(x, gamma):
            discount_cumsum = np.zeros_like(x)
            discount_cumsum[-1] = x[-1]
            for t in reversed(range(x.shape[0] - 1)):
                discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
            return discount_cumsum

        reward = {
            image_key: self.h5_file[f"{image_key}_{self.config.vl_type}_reward"][:, -1].astype(np.float32)
            for image_key in self.config.image_key.split(", ")
        }
        self.reward_min = {image_key: np.min(reward) for image_key, reward in reward.items()}
        self.reward_max = {image_key: np.max(reward) for image_key, reward in reward.items()}

        if self.config.use_normalize:
            modified_reward = {image_key: (reward - self.reward_min[image_key]) for image_key, reward in reward.items()}
        else:
            modified_reward = reward

        modified_pos_rtgs = {image_key: [] for image_key in self.config.image_key.split(", ")}
        for image_key in modified_pos_rtgs:
            _modified_pos_rtgs = modified_pos_rtgs[image_key]
            for idx in range(len(self.h5_file_traj_idx) - 1):
                stack = deque([], maxlen=self.config.num_frames)
                traj_indices = list(range(self.h5_file_traj_idx[idx], self.h5_file_traj_idx[idx + 1]))
                traj_cumsum = discount_cumsum(modified_reward[image_key][traj_indices], gamma=1.0)
                for i in range(len(traj_indices)):
                    if i == 0:
                        stack.extend([traj_cumsum[i]] * self.config.num_frames)
                    else:
                        stack.append(traj_cumsum[i])
                    _modified_pos_rtgs.append(np.stack(stack))
            _modified_pos_rtgs = np.asarray(_modified_pos_rtgs)

        # Automatically determine Return-to-go.
        if "coinrun" in self.env_name:
            self.return_to_go = np.max(list(modified_pos_rtgs.values())) // 100 * 100
        else:
            self.return_to_go = np.quantile(list(modified_pos_rtgs.values()), 0.9) // 100 * 100
        self.scale = compute_scale(self.return_to_go)
        self.config.scale = self.scale
        return modified_pos_rtgs

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def __getitem__(self, index):
        index = self.process_index(index)
        res = {"image": {}, "rtg": {}, "goal": {}}
        for key in self.config.image_key.split(", "):
            obs_frames = self.h5_file[key][index][-self.window_size :]

            # goal_frames to use hindsight relabeling.
            subsequent_indices = list(range(index, self.h5_file_traj_idx[self.idx_to_traj[index] + 1]))
            goal_indices = np.random.choice(subsequent_indices, 1).item()
            goal_frames = self.h5_file[key][min(goal_indices, self.h5_file["ob"].shape[0] - 1)][-self.window_size :]

            res["image"][key] = obs_frames
            res["goal"][key] = goal_frames
            if self.config.use_vl:
                if self.config.use_task_reward:
                    res["rtg"][key] = (
                        self.h5_file["rtg"][index][-self.window_size :][..., None]
                        - self.h5_file["rtg"][index][-self.window_size][..., None]
                    ) / self.config.scale
                else:
                    res["rtg"][key] = self.rtgs[key][index][-self.window_size :][..., None] / self.config.scale
        if self.config.state_key != "":
            res["state"] = np.concatenate(
                [self.h5_file[k][index] for k in self.config.state_key.split(", ")],
                axis=-1,
            )[-self.window_size :]

        res["action"] = self.h5_file["act"][index][-self.window_size :]
        instruct = get_m3ae_instruct(self.env_name)
        tokenized_instruct, padding_mask = self.tokenizer(instruct)
        res["instruct"] = tokenized_instruct
        res["text_padding_mask"] = padding_mask

        return res

    def build_tokenizer(self):
        use_bert_tokenizer = self.config.use_bert_tokenizer
        tokenizer_max_length = self.config.tokenizer_max_length

        if use_bert_tokenizer:
            import transformers

            tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            from .models.openai import tokenizer

            tokenizer = tokenizer.build_tokenizer()

        def tokenizer_fn(instruct):
            if use_bert_tokenizer:
                if len(instruct) == 0:
                    tokenized_instruct = np.zeros(tokenizer_max_length, dtype=np.int32)
                    padding_mask = np.ones(tokenizer_max_length, dtype=np.float32)
                else:
                    encoded_instruct = tokenizer(
                        instruct,
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer_max_length,
                        return_tensors="np",
                        add_special_tokens=False,
                    )
                    tokenized_instruct = encoded_instruct["input_ids"][0].astype(np.int32)
                    padding_mask = 1.0 - encoded_instruct["attention_mask"][0].astype(np.float32)
            else:
                tokenized_instruct = np.asarray(tokenizer(instruct)[0]).astype(np.int32)
                padding_mask = np.ones(tokenizer_max_length, dtype=np.float32)
            return tokenized_instruct, padding_mask

        return tokenizer_fn

    @property
    def num_actions(self):
        return self.config.action_dim

    @property
    def obs_shape(self):
        res = {"image": {}, "rtg": {}}
        for key in self.config.image_key.split(", "):
            res["image"][key] = (self.config.image_size, self.config.image_size, 3)
            res["rtg"][key] = (1,)
        if self.config.state_key != "":
            res["state"] = self.config.state_dim
        return res


def get_m3ae_instruct(task):
    if task == "coinrun":
        return f"the goal is to collect the coin."
    elif task == "coinrun_aisc":
        return f"the goal is to collect the coin."
    elif task == "maze":
        return f"navigate a maze to collect the yellow cheese."
    elif task == "maze_aisc":
        return f"navigate a maze to collect the yellow cheese."
    elif task == "maze_yellowline":
        return f"navigate a maze to collect the yellow line."
    elif task == "maze_redline_yellowgem":
        return f"navigate a maze to collect the red line."


def get_clip_instruct(task):
    if task == "coinrun":
        return f"the goal is to collect the coin."
    elif task == "coinrun_aisc":
        return f"the goal is to collect the coin."
    elif task == "maze":
        return f"navigate a maze to collect the yellow cheese."
    elif task == "maze_aisc":
        return f"navigate a maze to collect the yellow cheese."
    elif task == "maze_yellowline":
        return f"navigate a maze to collect the yellow line."
    elif task == "maze_redline_yellowgem":
        return f"navigate a maze to collect the red line."


def get_clip_special_instruct(env_name, inst_type):
    if inst_type == "random1":
        return "His voice echoed through the empty hallway."
    elif inst_type == "random2":
        return "NeurIPS 2023 will be held again at the at the New Orleans Ernest N. Morial Convention Center."
    elif inst_type == "misinfo":
        if "coinrun" in env_name:
            return "The agent must go to the far right of the level."
        elif env_name == "maze_aisc":
            return "navigate a maze to reacth to the top right corner."
        elif env_name == "maze_yellowline":
            return "navigate a maze to collect yellow gem."
    elif inst_type == "misinfo2":
        if "coinrun" in env_name:
            return "The goal is to collect the red strawberry."
    elif inst_type == "misinfo3":
        if "coinrun" in env_name:
            return "The goal is to reach the saw."
    elif inst_type == "misinfo4":
        if "coinrun" in env_name:
            return "The goal is to jump as high as you can."
    raise ValueError("You must pass any condition.")
