import argparse
import os

import h5py
import imageio
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm, trange


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


def generate_video(traj_dir, save_dir, save_type="hdf5", fps=20, write_index=False, skip_frame=1):
    if save_type == "npy":
        # DEPRECATED
        traj_list = os.listdir(traj_dir)
        for _traj in tqdm(traj_list):
            traj = np.load(os.path.join(traj_dir, _traj), allow_pickle=True).item()
            obs = traj["observations"]
            vr = VideoRecorder(save_dir=save_dir)
            for _ob in obs:
                vr.record(_ob[0, ...])
            _traj_filename = os.path.splitext(os.path.basename(_traj))[0]
            vr.save(f"video_{_traj_filename}.gif")

    elif save_type == "hdf5":
        h5_file = h5py.File(os.path.join(traj_dir, "data.hdf5"), "r")
        vr = VideoRecorder(save_dir=save_dir, fps=fps)
        num_episodes, ep_cursor = 0, 0

        for idx in trange(len(h5_file["ob"])):
            if idx % skip_frame == 0:
                if write_index:
                    frame_img = Image.fromarray(h5_file["ob"][idx][-1])
                    draw = ImageDraw.Draw(frame_img)
                    draw.text((0, 0), f"Timestep {ep_cursor + 1}", fill="yellow")
                    vr.record(np.asarray(frame_img))
                else:
                    vr.record(h5_file["ob"][idx][-1])
            ep_cursor += 1

            if h5_file["done"][idx][-1]:
                vr.save(f"video_{num_episodes}.gif")
                vr = VideoRecorder(save_dir=save_dir, fps=fps)
                num_episodes += 1
                ep_cursor = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, default="./demonstrations")
    parser.add_argument("--save_dir", type=str, default="./videos")
    parser.add_argument("--save_type", type=str, default="hdf5")
    parser.add_argument("--fps", type=int, default=20)
    parser.set_defaults(write_index=False)
    parser.add_argument("--write_index", action="store_true")

    args = parser.parse_args()
    traj_id = os.path.basename(args.traj_dir)
    save_dir = os.path.join(args.save_dir, traj_id)
    os.makedirs(save_dir, exist_ok=True)
    generate_video(args.traj_dir, save_dir, fps=args.fps, save_type=args.save_type, write_index=args.write_index)
