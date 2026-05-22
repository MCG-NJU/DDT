import copy
import timm
import torch
import torch.nn.functional as F

from typing import Callable
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler
from src.utils.no_grad import no_grad


def inverse_sigma(alpha, sigma):
    return 1 / sigma**2


def snr(alpha, sigma):
    return alpha / sigma


def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, min=threshold)


def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, max=threshold)


def constant(alpha, sigma):
    return 1


class DINOv2(nn.Module):
    def __init__(self, weight_path: str):
        super(DINOv2, self).__init__()
        self.encoder = torch.hub.load(
            '/mnt/bn/wangshuai6/torch_hub/facebookresearch_dinov2_main',
            weight_path,
            source="local",
            skip_validation=True,
        )
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()

    def fetch_pos(self, h, w):
        key = (h, w)
        if key in self.precomputed_pos_embed:
            return self.precomputed_pos_embed[key]
        value = timm.layers.pos_embed.resample_abs_pos_embed(self.pos_embed.data, [h, w])
        self.precomputed_pos_embed[key] = value
        return value

    def forward(self, x):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = F.interpolate(x, (int(224 * h / 256), int(224 * w / 256)), mode='bicubic')

        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h // self.patch_size[0], w // self.patch_size[1]
        self.encoder.pos_embed.data = self.fetch_pos(patch_num_h, patch_num_w)
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        return feature


def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)


class REPASelfConsistentDinoSLogOnlyTMinTrainer(BaseTrainer):
    def __init__(
        self,
        scheduler: BaseScheduler,
        loss_weight_fn: Callable = constant,
        fm_t_loss_weight: float = 1.0,
        fm_s_loss_weight: float = 1.0,
        dino_t_feat_loss_weight: float = 0.5,
        dino_s_feat_loss_weight: float = 0.5,
        lognorm_t=False,
        timeshift=1.0,
        encoder_weight_path=None,
        align_layer=8,
        proj_denoiser_dim=256,
        proj_hidden_dim=256,
        proj_encoder_dim=256,
        x_noised_std=0.0,
        v_mix_alpha=0.9,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.timeshift = timeshift
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn

        self.fm_t_loss_weight = fm_t_loss_weight
        self.fm_s_loss_weight = fm_s_loss_weight
        self.dino_t_feat_loss_weight = dino_t_feat_loss_weight
        self.dino_s_feat_loss_weight = dino_s_feat_loss_weight

        self.align_layer = align_layer
        self.encoder = DINOv2(encoder_weight_path)
        self.proj_encoder_dim = proj_encoder_dim
        no_grad(self.encoder)

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )

        self.x_noised_std = x_noised_std
        self.v_mix_alpha = v_mix_alpha

    def _resolve_hw(self, length, height, width):
        grid_h = int((length ** 0.5) * ((height / width) ** 0.5))
        grid_h = max(1, min(grid_h, length))
        while length % grid_h != 0 and grid_h > 1:
            grid_h -= 1
        grid_w = max(1, length // grid_h)
        return grid_h, grid_w

    def _match_token_resolution(self, src_feature, dst_feature, height, width):
        if dst_feature.shape[1] == src_feature.shape[1]:
            return dst_feature

        batch_size = src_feature.shape[0]
        dst_length = dst_feature.shape[1]
        dst_height, dst_width = self._resolve_hw(dst_length, height, width)
        src_height, src_width = self._resolve_hw(src_feature.shape[1], height, width)

        dst_feature = dst_feature.view(batch_size, dst_height, dst_width, self.proj_encoder_dim)
        dst_feature = dst_feature.permute(0, 3, 1, 2)
        dst_feature = F.interpolate(dst_feature, size=(src_height, src_width), mode='bilinear', align_corners=False)
        dst_feature = dst_feature.permute(0, 2, 3, 1).reshape(batch_size, -1, self.proj_encoder_dim)
        return dst_feature

    def _impl_trainstep(self, net, ema_net, raw_images, x, y):
        # x is clean latent (x1)
        x = x + torch.randn_like(x) * self.x_noised_std

        batch_size, c, height, width = x.shape
        # sample twice and reorder with min/max
        # t=min(t_a, s_a), s=max(t_a, s_a)
        if self.lognorm_t:
            base_a = torch.randn((batch_size), device=x.device, dtype=x.dtype).sigmoid()
            base_b = torch.randn((batch_size), device=x.device, dtype=x.dtype).sigmoid()
        else:
            base_a = torch.rand((batch_size), device=x.device, dtype=x.dtype)
            base_b = torch.rand((batch_size), device=x.device, dtype=x.dtype)
        t_a = shift_respace_fn(base_a, self.timeshift)
        s_a = shift_respace_fn(base_b, self.timeshift)
        t = torch.minimum(t_a, s_a)
        s = torch.maximum(t_a, s_a)

        noise = torch.randn_like(x)
        alpha_t = self.scheduler.alpha(t)
        dalpha_t = self.scheduler.dalpha(t)
        sigma_t = self.scheduler.sigma(t)
        dsigma_t = self.scheduler.dsigma(t)

        x_t = alpha_t * x + noise * sigma_t
        v_t_target = dalpha_t * x + dsigma_t * noise

        src_features = []

        def forward_hook(_, __, output):
            src_features.append(output)

        handle = net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        # t-branch prediction
        v_t_pred = net(x_t, t, y)
        src_t = self.proj(src_features[0])

        # build x_s from mixed velocity
        # v_mix = a * v_t_target + (1 - a) * v_t_pred
        v_mix = self.v_mix_alpha * v_t_target + (1.0 - self.v_mix_alpha) * v_t_pred
        x_s = x_t + (s - t)[:, None, None, None] * v_mix

        # s-branch prediction
        v_s_pred = net(x_s, s, y)
        src_s = self.proj(src_features[1])
        handle.remove()

        # requested formula with sign-safe clamp preserving denominator negativity
        # equivalent to (x1 - x_s) / (1 - s).clamp(min=0.05)
        x1 = x
        v_s_target = (x1 - x_s) / (1.0 - s).clamp(min=0.05)[:, None, None, None]

        with torch.no_grad():
            dst_feature = self.encoder(raw_images)
        dst_feature = self._match_token_resolution(src_t, dst_feature, height, width)

        # dino alignments for t and s branches
        dino_t_cos = 1.0 - F.cosine_similarity(src_t, dst_feature, dim=-1)
        dino_s_cos = 1.0 - F.cosine_similarity(src_s, dst_feature, dim=-1)

        weight_t = self.loss_weight_fn(alpha_t, sigma_t)
        fm_t_loss = weight_t * (v_t_pred - v_t_target) ** 2
        fm_s_loss = weight_t * (v_s_pred - v_s_target) ** 2

        feat_loss = (
            self.dino_t_feat_loss_weight * dino_t_cos.mean()
            + self.dino_s_feat_loss_weight * dino_s_cos.mean().detach()
        )

        total_loss = (
            self.fm_t_loss_weight * fm_t_loss.mean()
            + self.fm_s_loss_weight * fm_s_loss.mean()
            + feat_loss
        )

        return dict(
            fm_loss=fm_t_loss.mean(),
            fm_s_loss=fm_s_loss.mean(),
            cos_loss=dino_t_cos.mean(),
            dino_s_cos_loss=dino_s_cos.mean(),
            feat_loss=feat_loss,
            loss=total_loss,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars,
        )
