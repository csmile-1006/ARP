import clip
import numpy as np
import torch
from PIL import Image

from ..label_reward import center_crop

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_clip_reward(clip_model, obs, pos_text, use_crop=False):
    model, preprocess = clip_model
    if use_crop:
        obs = center_crop(obs[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
    image = preprocess(Image.fromarray(np.array(obs))).unsqueeze(0).to(device)
    text = clip.tokenize(pos_text).to(device)
    with torch.no_grad():
        _, logits_per_text = model(image, text)
    if isinstance(pos_text, list):
        clip_reward = logits_per_text.mean(axis=0)
    else:
        clip_reward = logits_per_text[0]
    return clip_reward.float().detach().cpu().numpy()


def get_torch_clip_goal_conditioned_reward(clip_model, obs, goal_image, use_crop=False):
    model, preprocess = clip_model
    if use_crop:
        obs = center_crop(obs[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
        goal_image = center_crop(goal_image[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
    image = preprocess(Image.fromarray(np.array(obs))).unsqueeze(0).to(device)
    goal_image = preprocess(Image.fromarray(np.array(goal_image))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image)
        goal_image_feature = model.encode_image(goal_image)

        # image_feature /= image_feature.norm(dim=1, keepdim=True)
        # goal_image_feature /= goal_image_feature.norm(dim=1, keepdim=True)

        goal_reward = -1 * torch.norm(image_feature - goal_image_feature).item()
        return goal_reward


def get_torch_clip_adapter_reward(clip_model, obs, pos_text, use_crop=False):
    model, preprocess = clip_model
    if use_crop:
        obs = center_crop(obs[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
    image = preprocess(Image.fromarray(np.array(obs))).unsqueeze(0).to(device)
    text = clip.tokenize(pos_text).to(device)
    with torch.no_grad():
        encoded_image = model.encode_image(image)
        encoded_text = model.encode_text(text)
        logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else model.clip_model.logit_scale.exp()
        logit = (logit_scale * (encoded_image @ encoded_text.T)).t()

    if isinstance(pos_text, list):
        clip_reward = logit.mean(axis=0)
    else:
        clip_reward = logit[0]

    return clip_reward.float().detach().cpu().numpy()


def get_torch_clip_adapter_goal_conditioned_reward(clip_model, obs, goal_image, use_crop=False):
    model, preprocess = clip_model
    if use_crop:
        obs = center_crop(obs[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
        goal_image = center_crop(goal_image[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
    image = preprocess(Image.fromarray(np.array(obs))).unsqueeze(0).to(device)
    goal_image = preprocess(Image.fromarray(np.array(goal_image))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image)
        goal_image_feature = model.encode_image(goal_image)

        # image_feature /= image_feature.norm(dim=1, keepdim=True)
        # goal_image_feature /= goal_image_feature.norm(dim=1, keepdim=True)

        goal_reward = -1 * torch.norm(image_feature - goal_image_feature).item()
        return goal_reward


def get_vip_reward(clip_model, obs, goal_image, use_crop=False):
    model, preprocess = clip_model
    if use_crop:
        obs = center_crop(obs[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
        goal_image = center_crop(goal_image[None, ...], (obs.shape[0] // 2, obs.shape[0] // 2))[0]
    image = preprocess(Image.fromarray(np.array(obs))).unsqueeze(0).to(device)
    goal_image = preprocess(Image.fromarray(np.array(goal_image))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model(image)
        goal_image_feature = model(goal_image)

        # image_feature /= image_feature.norm(dim=1, keepdim=True)
        # goal_image_feature /= goal_image_feature.norm(dim=1, keepdim=True)

        goal_reward = torch.norm(image_feature - goal_image_feature).item()
        return goal_reward
