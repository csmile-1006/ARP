import clip
import numpy as np
import torch
from einops import rearrange
from ml_collections import ConfigDict
from torch import nn
from torchvision.transforms.functional import normalize, resize


class Decoder(nn.Module):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.encoder_type = "clip"
        config.encoder_checkpoint = ""

        # Procgen specific config
        config.use_discrete_action = True
        config.action_dim = 15

        # Decoder Config
        config.latent_dim = 512
        config.image_size = 256

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config_updates=None, device: torch.device = None):
        super().__init__()
        self.config = self.get_default_config(config_updates)
        self.device = device

        if self.config.encoder_type == "clip":
            self.encoder, _ = clip.load("ViT-B/16", device=self.device)
        elif self.config.encoder_type == "clip_action_finetune":
            from finetune_module.clip_adapter import CLIPAdapter as CLIPActionAdapter

            self.encoder = CLIPActionAdapter(
                device=self.device,
                use_discrete_action=self.config.use_discrete_action,
                action_dim=self.config.action_dim,
            ).to(self.device)
            assert self.config.encoder_checkpoint != "", "You have to specifiy vl_checkpoint."
            model_state_dict = torch.load(self.config.encoder_checkpoint, map_location=self.device)
            self.encoder.load_state_dict(model_state_dict)
            self.encoder.eval()

        self.shape = 224 // 4
        self.latent_fc = nn.Linear(self.config.latent_dim, (self.shape**2) * 32)
        self.conv1 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()

        # CLIP Statistics
        self.image_mean = (0.48145466, 0.4578275, 0.40821073)
        self.image_std = (0.26862954, 0.26130258, 0.27577711)

    def decode(self, latent):
        x = self.relu(self.latent_fc(latent))
        x = torch.reshape(x, (x.shape[0], 32, self.shape, self.shape))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def preprocess(self, x):
        B, H, W, C = x.shape
        assert C == 3
        x = rearrange(x, "b h w c -> b c h w")
        x = resize(x, (224, 224)) if H != 224 and W != 224 else x
        # this is a rgb image, just normalize
        x = x.float() / 255.0
        # convert to BCTHW
        x = normalize(x, mean=self.image_mean, std=self.image_std)
        return x

    def forward(self, batch):
        image_batch = batch["image1"]
        images = torch.stack(list(image_batch.values()))
        images = images.reshape((-1, *images.shape[-3:]))
        preprocessed_images = self.preprocess(images)
        latent = self.encoder.encode_image(preprocessed_images).float()
        recon_images = self.decode(latent)
        loss = self.loss(recon_images, preprocessed_images)
        return loss

    def recon_image(self, image: np.ndarray):
        image = torch.from_numpy(image).to(self.device).unsqueeze(0)
        preprocessed_image = self.preprocess(image)
        latent = self.encoder.encode_image(preprocessed_image).float()
        recon_image = self.decode(latent).squeeze(0)
        recon_image = normalize(
            recon_image,
            mean=[-_mean / _std for _mean, _std in zip(self.image_mean, self.image_std)],
            std=[1 / _std for _std in self.image_std],
        )
        recon_image = (recon_image * 255.0).permute(1, 2, 0).long().detach().cpu().numpy().astype(np.uint8)
        return recon_image
