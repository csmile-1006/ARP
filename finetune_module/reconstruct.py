# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import os
import random

import absl
import numpy as np
import torch
from absl import app
from MUGEN_baseline.retrieval.utils import AvgMeter, get_lr
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arp_dt.envs.procgen import Procgen
from arp_dt.utils import define_flags_with_default, get_user_flags

from .action_finetune_data_metaworld import MetaWorldActionDataset
from .action_finetune_data_procgen import ProcgenActionDataset
from .decoder import Decoder

FLAGS_DEF = define_flags_with_default(
    seed=0,
    model=Decoder.get_default_config(),
    data=ProcgenActionDataset.get_default_config(),
    env=Procgen.get_default_config(),
    model_name="v2t",
    batch_size=32,
    num_workers=8,
    game_name="coinrun",
    model_path="",
    default_root_dir="./saved_checkpoints",
    lr=5e-4,
    weight_decay=0.001,
    epochs=100,
    temperature=0.07,
    max_temperature=100.0,
    video_enc=True,
    audio_enc=False,
    text_enc=True,
    pretrained=True,
    trainable=True,
    model_type="clip",
)
FLAGS = absl.flags.FLAGS


def build_loaders(config, dataset_name, dataset_class):
    # dataset = MUGENDataset(args, split)
    dataset = dataset_class(update=config, dataset_name=dataset_name)
    num_train = int(len(dataset) * 0.9)
    num_valid = len(dataset) - num_train
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        shuffle=True,
        drop_last=True,
        # persistent_workers=True,
        # prefetch_factor=2
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        shuffle=False,
        drop_last=True,
        # persistent_workers=True,
        # prefetch_factor=2
    )
    return train_loader, valid_loader


def train_epoch(model, device, train_loader, optimizer, writer, n_iter):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        for k, v in batch.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    batch[k][_k] = _v.to(device)
            else:
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = v
        loss = model(batch)

        writer.add_scalar("Loss/train", loss.item(), n_iter)
        # writer.add_scalar("img_acc/train", img_acc.item(), n_iter)
        # writer.add_scalar("text_acc/train", text_acc.item(), n_iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch["action"].size(0)
        loss_meter.update(loss.item(), count)
        n_iter += 1
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter, n_iter


@torch.no_grad()
def valid_epoch(model, device, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        for k, v in batch.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    batch[k][_k] = _v.to(device)
            else:
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = v
        loss = model(batch)
        # loss = loss_fn(score_1, score_2, batch["action"]).mean()

        count = batch["action"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter.avg


def train(_):
    args = get_user_flags(FLAGS, FLAGS_DEF)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    os.makedirs(FLAGS.default_root_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=FLAGS.default_root_dir, comment=f"--{FLAGS.model_name}")
    # valid_loader = build_loaders(args, "val")

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    if "v2" in FLAGS.game_name:
        dataset_name = FLAGS.game_name
        dataset_class = MetaWorldActionDataset
        conf = MetaWorldActionDataset.get_default_config()
        conf.path = FLAGS.data.path
        FLAGS.data = conf
        FLAGS.model.use_discrete_action = False
    else:
        dataset_name = f"{FLAGS.game_name}_{FLAGS.env.distribution_mode}_level{FLAGS.env.start_level}to{FLAGS.env.num_levels}_num{FLAGS.data.num_demonstrations}_frame{FLAGS.data.num_frames}"
        if FLAGS.data.train_env_type != "none":
            dataset_name += f"_{FLAGS.data.train_env_type}"
        dataset_class = ProcgenActionDataset
        FLAGS.model.use_discrete_action = True
    train_loader, valid_loader = build_loaders(FLAGS.data, dataset_name, dataset_class)

    # if FLAGS.model_type == "mugen":
    #     model = MUGENAdapter(
    #         model_path=FLAGS.model_path,
    #         use_discrete_action=use_discrete_action
    #     ).to(device)
    # elif FLAGS.model_type == "clip":
    #     model = CLIPAdapter(
    #         device=device,
    #         use_discrete_action=use_discrete_action,
    #         action_dim=FLAGS.data.action_dim
    #     ).to(device)
    FLAGS.model.encoder_type = FLAGS.model_type
    FLAGS.model.encoder_checkpoint = FLAGS.model_path
    model = Decoder(config_updates=FLAGS.model).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    best_loss = float("inf")
    n_iter = 0
    for epoch in range(FLAGS.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, n_iter = train_epoch(model, device, train_loader, optimizer, writer, n_iter)

        model.eval()
        valid_loss = valid_epoch(model, device, valid_loader)
        # valid_loss, img_acc, text_acc = valid_epoch(model, valid_loader)

        writer.add_scalar("Loss/val", valid_loss, epoch)
        # writer.add_scalar("img_acc/val", img_acc, epoch)
        # writer.add_scalar("text_acc/val", text_acc, epoch)
        # torch.save(model.state_dict(), os.path.join(FLAGS.default_root_dir, f"epoch={epoch}.pt"))

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(FLAGS.default_root_dir, "best_checkpoint.pt"))
            print("Saved Best Model!")

    writer.close()


if __name__ == "__main__":
    app.run(train)
