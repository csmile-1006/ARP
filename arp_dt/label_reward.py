import argparse
import os
from collections import deque

import clip
import h5py
import numpy as np
import torch
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToPILImage, ToTensor
from tqdm import trange

from arp_dt.data_procgen import get_clip_instruct, get_clip_special_instruct


def center_crop(image, crop_size):
    """
    Performs center cropping on an image.

    Args:
        image (np.ndarray): Input image vector with shape [N, H, W, C].
        crop_size (tuple): Tuple containing the desired crop size (height, width).

    Returns:
        np.ndarray: Cropped image vector with shape [N, crop_height, crop_width, C].
    """
    N, H, W, C = image.shape
    crop_height, crop_width = crop_size

    # Calculate crop starting positions
    start_h = int((H - crop_height) / 2)
    start_w = int((W - crop_width) / 2)

    # Perform center cropping
    cropped_image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

    return cropped_image


def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


def label_reward(
    env_name,
    distribution_mode,
    num_levels,
    start_level,
    text,
    base_path,
    data_path=None,
    image_keys="ob",
    num_demonstrations=500,
    num_frames=8,
    env_type=None,
    model_type="clip",
    model_ckpt_dir=None,
    use_crop=False,
    inst_type="none",
):
    image_keys = image_keys.split(", ")
    if data_path is None:
        dirname = (
            f"{env_name}_{distribution_mode}_level{start_level}to{num_levels}_num{num_demonstrations}_frame{num_frames}"
        )
        if env_type != "none":
            dirname += f"_{env_type}"
        data_path = os.path.join(base_path, dirname, "data.hdf5")
    g = h5py.File(data_path, "a")

    if g.get("done"):
        done_key = "done"
    elif g.get("rewards"):
        done_key = "rewards"
    elif g.get("is_terminal"):
        done_key = "is_terminal"
    else:
        raise ValueError

    try:
        len_data, num_frames = g[done_key].shape[:2]
        g_traj_idx = list(np.nonzero(g[done_key][:, -1])[0] + 1)
        g_traj_idx.insert(0, 0)
    except:
        len_data, num_frames = g["time"].shape[:2]
        g_traj_idx = list(np.where(g["time"][:, -1, 0] == 1.0)[0])
        g_traj_idx.append(len(g["time"]))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if use_crop:

        def _transform(image_size, n_px=224):
            return Compose(
                [
                    ToPILImage(),
                    CenterCrop(image_size // 2),
                    Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )

        image_size = g[image_keys[0]].shape[-2]
        print(f"image_size: {image_size}")
        transform = _transform(image_size)
    else:

        def _transform(n_px=224):
            return Compose(
                [
                    ToPILImage(),
                    Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(n_px),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )

        transform = _transform()

    if model_type.startswith("clip"):

        def load_model():
            return clip.load("ViT-B/16", device=device)

        if model_type == "clip":
            model, _ = load_model()
            clip_model = (model, transform)

            def compute_reward(clip_model, images, text=text):
                model, preprocess = clip_model
                images = torch.from_numpy(np.stack([preprocess(img) for img in images])).to(device)
                if isinstance(text, list):
                    text = clip.tokenize(text).to(device)
                else:
                    text = clip.tokenize([text]).to(device)

                with torch.no_grad():
                    _, logits_per_text = model(images, text)
                    if isinstance(text, list):
                        clip_reward = logits_per_text.mean(axis=0)
                    else:
                        clip_reward = logits_per_text[0]
                return clip_reward.cpu().numpy()

        elif model_type == "clip_goal_conditioned":
            model, _ = load_model()
            clip_model = (model, transform)

            def compute_reward(clip_model, images, text=text):
                model, preprocess = clip_model
                images = torch.from_numpy(np.stack([preprocess(img) for img in images])).to(device)
                with torch.no_grad():
                    images_feature = model.encode_image(images)
                    # images_feature /= images_feature.norm(dim=1, keepdim=True)

                    goal_image_feature = images_feature[-1]
                    goal_reward = -1 * np.array(
                        [torch.norm(image_feat - goal_image_feature, p=2).item() for image_feat in images_feature]
                    )
                return goal_reward

        elif model_type.startswith("clip_"):
            if model_type == "clip_ft":
                from finetune_module.clip_multiscale_adapter import CLIPMultiscaleAdapter

                model = CLIPMultiscaleAdapter(
                    device=device,
                    use_discrete_action=True,
                    action_dim=15,
                ).to(device)
            assert model_ckpt_dir is not None, "specify model_ckpt_dir"
            model_state_dict = torch.load(model_ckpt_dir)
            model.load_state_dict(model_state_dict, strict=False)
            model.eval()
            clip_model = (model, None)

            if "_goal_conditioned" in model_type:

                def compute_reward(clip_model, images, text=text):
                    model, _ = clip_model
                    if use_crop:
                        images = center_crop(images, crop_size=(image_size // 2, image_size // 2))
                    with torch.no_grad():
                        images_feature = model.encode_image(
                            model.preprocess(torch.from_numpy(images).to(device), train=False)
                        )
                        # images_feature /= images_feature.norm(dim=1, keepdim=True)

                        goal_image_feature = images_feature[-1]
                        goal_reward = np.array(
                            [torch.norm(image_feat - goal_image_feature, p=2).item() for image_feat in images_feature]
                        )
                    return goal_reward

            else:

                def compute_reward(clip_model, images, text=text):
                    model, _ = clip_model
                    if use_crop:
                        images = center_crop(images, crop_size=(image_size // 2, image_size // 2))
                    if isinstance(text, list):
                        text = clip.tokenize(text).to(device)
                    else:
                        text = clip.tokenize([text]).to(device)
                    with torch.no_grad():
                        images_feature = model.encode_image(
                            model.preprocess(torch.from_numpy(images).to(device), train=False)
                        )
                        pos_seq_output = model.encode_text(text)
                        logit_scale = (
                            model.clip_model.logit_scale if not hasattr(model, "logit_scale") else model.logit_scale
                        )
                        logit_scale = logit_scale.exp()
                        if len(images_feature.shape) == 3:
                            images_feature = images_feature.reshape(len(images), -1)
                            pos_seq_output = pos_seq_output.reshape(text.shape[0], -1)
                            logit = (logit_scale * (images_feature @ pos_seq_output.T)).t() / (
                                model.num_clip_layers + 1
                            )
                        else:
                            logit = (logit_scale * (images_feature @ pos_seq_output.T)).t()
                    if isinstance(text, list):
                        clip_reward = logit.mean(axis=0)
                    else:
                        clip_reward = logit[0]

                    return clip_reward.float().detach().cpu().numpy()

    def stack_outputs(pos_outputs):
        # Re-stacking for aligning with original data.
        if len(pos_outputs.shape) == 0:
            pos_outputs = pos_outputs[None, ...]
        stacked_pos_outputs = []
        pos_stack = deque([], maxlen=num_frames)
        for i in trange(len(pos_outputs), desc="stacking again.", leave=False):
            if i == 0:
                pos_stack.extend([pos_outputs[i]] * num_frames)
            else:
                pos_stack.append(pos_outputs[i])
            stacked_pos_outputs.append(list(pos_stack))

        return np.asarray(stacked_pos_outputs)

    def discount_cumsum(x, gamma=1.0):
        if len(x.shape) == 0:
            x = x[None, ...]
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    # main part.
    target_keys = [f"{model_type}_reward", f"{model_type}_pos_rtg"]
    if inst_type != "none":
        target_keys = list(map(lambda x: f"{x}_{inst_type}", target_keys))
    for img_key in image_keys:
        pre_defined = {}
        for _key in target_keys:
            key = f"{img_key}_{_key}"
            pre_defined[key] = g.get(key)
        for idx in trange(len(g_traj_idx) - 1):
            data = {}
            traj = list(range(g_traj_idx[idx], min(g_traj_idx[idx + 1], len_data)))
            pos_outputs = compute_reward(clip_model, g[img_key][traj, -1], text=text)
            pos_rtg = discount_cumsum(pos_outputs)
            data[target_keys[0]] = stack_outputs(pos_outputs)
            data[target_keys[1]] = stack_outputs(pos_rtg)

            for _key in target_keys:
                key = f"{img_key}_{_key}"
                if not pre_defined[key]:
                    if idx == 0:
                        g.create_dataset(
                            key,
                            compression="gzip",
                            chunks=(1, num_frames),
                            maxshape=(len_data, num_frames),
                            data=data[_key],
                        )
                    else:
                        _dataset = g[key]
                        _dataset.resize((_dataset.shape[0] + data[_key].shape[0]), axis=0)
                        _dataset[-data[_key].shape[0] :] = data[_key]
                else:
                    g[key][traj] = data[_key]

    g.close()


def main():
    parser = argparse.ArgumentParser(description="Process rollout training arguments.")
    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--env_type", type=str, default="none")
    parser.add_argument("--num_levels", type=int, default=500)
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--distribution_mode", type=str, default="hard")
    parser.add_argument("--image_keys", type=str, default="ob")
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--base_path", type=str, default="./demonstrations")
    parser.add_argument("--num_demonstrations", type=int, default=500)
    parser.add_argument("--save_type", type=str, default="npy", choices=["npy", "hdf5"])
    parser.add_argument("--num_frames", type=int, default=8)

    parser.add_argument("--model_type", type=str, default="clip")
    parser.add_argument("--model_ckpt_dir", type=str, default=None)
    parser.add_argument("--use_crop", type=bool, default=False)
    parser.add_argument("--inst_type", type=str, default="none")

    args = parser.parse_args()
    env_name = f"{args.env_name}" if args.env_type == "none" else f"{args.env_name}_{args.env_type}"
    if args.inst_type != "none":
        text = get_clip_special_instruct(env_name, args.inst_type)
    else:
        text = get_clip_instruct(env_name)

    print(f"[INFO] env_name: {env_name}\t instruction: {text}")

    label_reward(
        env_name=args.env_name,
        env_type=args.env_type,
        distribution_mode=args.distribution_mode,
        image_keys=args.image_keys,
        data_path=args.data_path,
        text=text,
        num_levels=args.num_levels,
        start_level=args.start_level,
        num_demonstrations=args.num_demonstrations,
        num_frames=args.num_frames,
        base_path=args.base_path,
        model_type=args.model_type,
        model_ckpt_dir=args.model_ckpt_dir,
        use_crop=args.use_crop,
        inst_type=args.inst_type,
    )


if __name__ == "__main__":
    main()
