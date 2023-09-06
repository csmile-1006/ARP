import argparse
import os

import imageio
import numpy as np
from tqdm import tqdm


class VideoRecorder:
    def __init__(self, save_dir, fps=20):
        self.save_dir = save_dir
        self.fps = fps
        self.frames = []

    def init(self, frame):
        self.frames = []
        self.record(frame)

    def record(self, frame):
        self.frames.append(frame)

    def save(self, file_name):
        path = self.save_dir + "/" + file_name
        imageio.mimsave(str(path), self.frames, fps=self.fps)


def generate_video(traj_dir, save_dir):
    traj_list = os.listdir(traj_dir)
    for _traj in tqdm(traj_list):
        traj = np.load(os.path.join(traj_dir, _traj), allow_pickle=True).item()
        obs = traj["observations"]
        vr = VideoRecorder(save_dir=save_dir)
        for _ob in obs:
            vr.record(_ob[0, ...])
        _traj_filename = os.path.splitext(os.path.basename(_traj))[0]
        vr.save(f"video_{_traj_filename}.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, default="./demonstrations")
    parser.add_argument("--save_dir", type=str, default="./videos")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    generate_video(args.traj_dir, args.save_dir)
