import clip
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch import nn
from torchvision.transforms.functional import normalize, resize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from kornia.augmentation import ColorJitter

from .layers import AdapterMLP


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = nn.Sequential(ColorJitter(0.1, 0.2, 0.2, 0.03, same_on_batch=True, p=0.75))

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


class CLIPAdapter(nn.Module):
    def __init__(
        self,
        model_path: str = None,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        action_dim: int = 15,
        num_layers: int = 2,
        device: torch.device = None,
        use_discrete_action: bool = False,
        use_vip_loss: bool = False,
        use_id_loss: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        self.load_clip()
        # self.clip_model = CLIPModel(audio_enc=False, image_enc=True, text_enc=True)
        # self.clip_model.load_state_dict(torch.load(self.model_path))
        # self.clip_model.eval()
        self.augmentation = DataAugmentation()
        self.device = device

        self.use_vip_loss = use_vip_loss
        self.use_id_loss = use_id_loss

        self.image_adapter = AdapterMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers
        )

        self.text_adapter = AdapterMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers
        )

        self.inverse_layer = AdapterMLP(
            input_dim=4 * output_dim, hidden_dim=hidden_dim, output_dim=action_dim, num_layers=num_layers
        )

        self.image_residual_weight = nn.Parameter(torch.tensor(4.0))
        self.text_residual_weight = nn.Parameter(torch.tensor(4.0))

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = self.clip_model.logit_scale.detach().clone()
        self.use_discrete_action = use_discrete_action
        if self.use_discrete_action:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        if self.use_id_loss:
            # self.lambda_id = 0.1
            self.lambda_id = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gamma = 0.98

    def load_clip(self):
        self.clip_model, _ = clip.load("ViT-B/16")
        self.tokenizer = clip.tokenize

    def preprocess(self, x, train=False):
        B, H, W, C = x.shape
        assert C == 3
        x = rearrange(x, "b h w c -> b c h w")
        x = x.float()
        x = resize(x, (224, 224)) if H != 224 and W != 224 else x
        if train:
            x = self.augmentation(x)
        # this is a rgb image, just normalize
        x = x / 255.0
        # convert to BCTHW
        x = normalize(x, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        return x

    def encode_image(self, image):
        image_feature = self.clip_model.encode_image(image).float()

        res = torch.sigmoid(self.image_residual_weight)
        adapted_image = res * image_feature + (1.0 - res) * self.image_adapter(image_feature)
        adapted_image = F.normalize(adapted_image, dim=-1)
        return adapted_image

    def encode_text(self, text):
        text_shape = text.shape
        if len(text_shape) == 3:
            batch_size, num_text, ctx = text_shape
            text = text.reshape(-1, ctx)
        else:
            batch_size, ctx = text_shape
        text_feature = self.clip_model.encode_text(text).float()

        res = torch.sigmoid(self.text_residual_weight)
        adapted_text = res * text_feature + (1.0 - res) * self.text_adapter(text_feature)
        adapted_text = F.normalize(adapted_text, dim=-1)
        if len(text_shape) == 3:
            adapted_text = adapted_text.reshape(batch_size, num_text, -1)
            adapted_text = adapted_text.mean(dim=1)
        return adapted_text

    def tcn_distance(self, x1, x2):
        diff = torch.abs(x1 - x2)
        if len(diff.shape) > 1:
            return torch.sum(torch.pow(diff, 2), dim=-1)
        else:
            return torch.pow(diff, 2)

    def tcn_loss(self, feat1, feat2, feat3):
        d_positive = self.tcn_distance(feat1, feat2)
        d_negative = self.tcn_distance(feat1, feat3)
        loss = torch.mean(torch.clip(self.tcn_margin + d_positive - d_negative, min=0.0, max=torch.inf))
        return loss

    def forward(self, batch):
        image_batch_0, image_batch_1, image_batch_2, image_batch_3 = (
            batch["image0"],
            batch["image1"],
            batch["image2"],
            batch["image3"],
        )

        total_loss = 0.0
        for image_key in image_batch_1.keys():
            batch_size = image_batch_1[image_key].shape[0]
            total_image = torch.cat(
                [
                    image_batch_0[image_key],
                    image_batch_1[image_key],
                    image_batch_2[image_key],
                    image_batch_3[image_key],
                ],
                axis=0,
            )
            augmented_image = self.preprocess(total_image, train=True)
            augmented_image_0, augmented_image_1, augmented_image_2, augmented_image_3 = torch.split(
                augmented_image, batch_size
            )

            adapted_image_0, adapted_image_1, adapted_image_2, adapted_image_3 = (
                self.encode_image(augmented_image_0),
                self.encode_image(augmented_image_1),
                self.encode_image(augmented_image_2),
                self.encode_image(augmented_image_3),
            )
            adapted_text = self.encode_text(batch["instruct"])

            # VIP Loss
            logit_scale = self.logit_scale.exp()
            score_0 = torch.diag(logit_scale * (adapted_image_0 @ adapted_text.T), 0)
            score_1 = torch.diag(logit_scale * (adapted_image_1 @ adapted_text.T), 0)
            score_2 = torch.diag(logit_scale * (adapted_image_2 @ adapted_text.T), 0)
            score_3 = torch.diag(
                logit_scale * (adapted_image_3 @ adapted_text.T), 0
            )  # score_1 = torch.mul(torch.diag((_adapted_image_1 @ _adapted_text.T), 0).reshape(batch_size, -1), logit_scale).mean(dim=-1)

            # VIP-I Loss
            r = batch["r"] - 1
            epsilon = 1e-8
            # vip_loss = (1 - self.gamma) * score_0.mean() + torch.logsumexp(score_1 + 1 - self.gamma * score_2)
            vip_loss = (1 - self.gamma) * -score_0.mean() + torch.log(
                epsilon + torch.mean(torch.exp(-(r + self.gamma * score_2 - score_1)))
            )
            # V_loss = (1-model.module.gamma) * -V_0.mean() + torch.log(epsilon + torch.mean(torch.exp(-(r + model.module.gamma * V_s_next - V_s))))

            # ID Loss
            image_1_feature = torch.concat([adapted_image_1, adapted_text], dim=-1)
            image_2_feature = torch.concat([adapted_image_2, adapted_text], dim=-1)

            concat_feature = torch.concat([image_1_feature, image_2_feature], dim=-1)
            action_output = self.inverse_layer(concat_feature)
            id_loss = self.loss_fn(action_output, batch["action"])

            if self.use_vip_loss:
                total_loss += vip_loss

            if self.use_id_loss:
                total_loss += self.lambda_id * id_loss

        return total_loss.mean()


if __name__ == "__main__":
    device = torch.device("cuda")
    model = CLIPAdapter(
        model_path=None, device=device, use_discrete_action=True, use_vip_loss=True, use_id_loss=True
    ).to(device)

    import h5py

    f = h5py.File(
        "/home/changyeon/procgen_generalization/procgen_data/clip/new_demonstrations_clip_ver3/coinrun_hard_level0to500_num500_frame8/data_train.hdf5",
        "r",
    )
    image = f["ob"][0, -1]

    image_feature = model.encode_image(model.preprocess(torch.Tensor(image).unsqueeze(0).to(device)))
    print(f"image_feature: {image_feature.shape}")

    text = ["Ive.", "NewJeans.", "Le serafim."]
    _text = clip.tokenize(text).to(device)
    text_feature = model.encode_text(_text)
    print(f"text_feature: {text_feature.shape}")

    batch = {
        "image0": {"ob": torch.from_numpy(image[None, ...]).to(device)},
        "image1": {"ob": torch.from_numpy(image[None, ...]).to(device)},
        "image2": {"ob": torch.from_numpy(image[None, ...]).to(device)},
        "image3": {"ob": torch.from_numpy(image[None, ...]).to(device)},
        "action": torch.rand(1, 15).to(device),
        "instruct": _text.unsqueeze(0),
        "r": torch.zeros(1).to(device),
    }

    loss = model(batch)
    print(f"loss: {loss.item()}")
