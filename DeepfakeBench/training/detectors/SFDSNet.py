import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class FTA(nn.Module):
  
    def __init__(
        self,
        embed_dim=1024,
        hidden_dim=256,
        spec_hidden=16,
        num_bands=4,
        dropout=0.1,
        max_token_residual=0.20,
        max_cls_residual=0.15,
        band_dropout=0.10,
        use_phase_gate=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.band_dropout = band_dropout
        self.use_phase_gate = use_phase_gate
        self.max_token_residual = max_token_residual
        self.max_cls_residual = max_cls_residual

        self.token_norm = nn.LayerNorm(embed_dim)
        self.cls_norm = nn.LayerNorm(embed_dim)

        # amplitude branch
        self.amp_conv = nn.Sequential(
            nn.Conv2d(embed_dim, spec_hidden, kernel_size=3, padding=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(spec_hidden, embed_dim, kernel_size=3, padding=1, groups=1, bias=False),
        )

        # phase descriptor branch
        self.phase_conv = nn.Sequential(
            nn.Conv2d(embed_dim, spec_hidden, kernel_size=3, padding=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(spec_hidden, embed_dim, kernel_size=3, padding=1, groups=1, bias=False),
        )

        # band attention
        band_in_dim = num_bands * (2 if use_phase_gate else 1)
        self.band_mlp = nn.Sequential(
            nn.Linear(band_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bands),
        )

        # token-level fusion gate
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid(),
        )

        # cls fusion
        self.cls_fuse = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # output projection
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # identity-like init
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

        self.token_alpha_logit = nn.Parameter(torch.tensor(-3.5))
        self.cls_alpha_logit = nn.Parameter(torch.tensor(-4.0))

        self.band_prior = nn.Parameter(torch.linspace(0.95, 1.10, num_bands))

    def _build_band_masks(self, H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        rr = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2)
        rr = rr / rr.max().clamp_min(1e-6)

        edges = torch.linspace(0.0, 1.0, self.num_bands + 1, device=device)
        masks = []
        for i in range(self.num_bands):
            low, high = edges[i], edges[i + 1]
            if i == self.num_bands - 1:
                m = ((rr >= low) & (rr <= high)).float()
            else:
                m = ((rr >= low) & (rr < high)).float()
            masks.append(m)
        return torch.stack(masks, dim=0)  # [K, H, W]

    def _band_pool(self, x, masks):
        # x: [B, C, H, W]
        pooled = []
        for k in range(self.num_bands):
            mk = masks[k][None, None, :, :]
            denom = mk.sum().clamp_min(1.0)
            pooled_k = (x * mk).sum(dim=(2, 3)) / denom  # [B, C]
            pooled_k = pooled_k.mean(dim=1)              # [B]
            pooled.append(pooled_k)
        return torch.stack(pooled, dim=1)  # [B, K]

    def _band_dropout(self, w):
        if (not self.training) or self.band_dropout <= 0:
            return w
        mask = (torch.rand_like(w) < self.band_dropout).float()
        uniform = torch.full_like(w, 1.0 / self.num_bands)
        w = w * (1.0 - mask) + uniform * mask
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return w

    def _band_reg(self, w):
        eps = 1e-8
        uniform = torch.full_like(w, 1.0 / self.num_bands)
        kl = (w * ((w + eps).log() - (uniform + eps).log())).sum(dim=1).mean()
        return kl

    def forward(self, cls_token, patch_tokens, return_aux=False):
        """
        cls_token:    [B, C]
        patch_tokens: [B, N, C]
        """
        B, N, C = patch_tokens.shape
        grid = int(math.sqrt(N))
        assert grid * grid == N, "Number of patch tokens must be a square"

        cls_res = cls_token
        tok_res = patch_tokens

        patch_tokens = self.token_norm(patch_tokens)
        cls_token = self.cls_norm(cls_token)

        # [B, N, C] -> [B, C, H, W]
        x = patch_tokens.transpose(1, 2).contiguous().view(B, C, grid, grid)

        # FFT over spatial grid
        spec = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        log_mag = torch.log1p(mag)

        masks = self._build_band_masks(grid, grid, x.device)

        mag_band = self._band_pool(log_mag, masks)   # [B, K]
        if self.use_phase_gate:
            phase_band = self._band_pool(phase.abs(), masks)
            band_input = torch.cat([mag_band, phase_band], dim=1)
        else:
            band_input = mag_band

        band_logits = self.band_mlp(band_input)
        band_w = torch.softmax(band_logits, dim=1)
        band_w = self._band_dropout(band_w)

        # build spectral weight map
        spec_weight = torch.einsum(
            'bk,khw->bhw',
            band_w * self.band_prior.unsqueeze(0),
            masks
        ).unsqueeze(1)  # [B,1,H,W]

        spec_weight = spec_weight / spec_weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        # amplitude enhancement
        amp_delta = torch.tanh(self.amp_conv(log_mag))
        if self.use_phase_gate:
            phase_gate = torch.sigmoid(self.phase_conv(phase.abs()))
        else:
            phase_gate = 1.0

        log_mag_new = log_mag + spec_weight * amp_delta * phase_gate

        # keep phase stable
        mag_new = torch.expm1(log_mag_new.clamp_min(0.0)).clamp_min(0.0)
        spec_new = torch.polar(mag_new + 1e-6, phase)

        # iFFT
        x_freq = torch.fft.ifft2(torch.fft.ifftshift(spec_new, dim=(-2, -1)), norm='ortho').real

        # back to tokens
        tok_freq = x_freq.view(B, C, N).transpose(1, 2).contiguous()  # [B, N, C]

        # token residual fusion
        token_gate = self.token_gate(torch.cat([tok_res, tok_freq], dim=-1))
        token_alpha = self.max_token_residual * torch.sigmoid(self.token_alpha_logit)
        tok_out = tok_res + token_alpha * token_gate * (tok_freq - patch_tokens)

        # token aggregation
        tok_global_mean = tok_out.mean(dim=1)
        tok_global_max = tok_out.max(dim=1)[0]

        # cls enhancement
        cls_alpha = self.max_cls_residual * torch.sigmoid(self.cls_alpha_logit)
        cls_delta = self.cls_fuse(torch.cat([cls_token, tok_global_mean, tok_global_max], dim=1))
        cls_out = cls_res + cls_alpha * cls_delta

        # final global feature
        global_feat = cls_out + self.out_proj(tok_global_mean)
        global_feat = F.layer_norm(global_feat, global_feat.shape[1:])

        aux = {
            'band_weights': band_w,
            'freq_reg': self._band_reg(band_w),
            'token_alpha': token_alpha.detach(),
            'cls_alpha': cls_alpha.detach(),
        }

        return (global_feat, aux) if return_aux else global_feat

class DSFE(nn.Module):

    def __init__(self, feature_dim=1024, subspace_dims=[256, 256, 512],
                 dropout=0.1, reduction=16, use_enhancer=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.subspace_dims = subspace_dims
        self.num_subspaces = len(subspace_dims)
        self.use_enhancer = use_enhancer

        assert sum(subspace_dims) == feature_dim, "subspace_dims sum must equal feature_dim"

        # 1. 输入投影
        self.project_in = nn.Linear(feature_dim, feature_dim)

        # 2. 子空间独立 MLP（Bottleneck 双层 + Dropout）
        self.subspace_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d // 2, d),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for d in subspace_dims
        ])

        # 3. 动态 residual scale MLP
        self.scale_mlp = nn.Sequential(
            nn.Linear(feature_dim, self.num_subspaces),
            nn.Sigmoid()  # 输出每个子空间的 scale
        )

        # 4. 子空间交互
        self.inter_subspace = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

        # 5. 全局 gating（增加高阶统计）
        # mean/std/max/min/skew/kurt -> 6 个额外维度
        self.gate_fc1 = nn.Linear(feature_dim + 6, feature_dim // reduction)
        self.gate_fc2 = nn.Linear(feature_dim // reduction, self.num_subspaces)

        # 6. 融合
        self.fusion = nn.Linear(feature_dim, feature_dim)

        # 7. LayerNorm
        self.norm = nn.LayerNorm(feature_dim)

        # 8. 跳跃连接
        self.skip_connection = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, return_gating=False):
        if not self.use_enhancer:
            return (x, None) if return_gating else x

        B = x.size(0)

        # 投影 + LayerNorm
        x_proj = self.project_in(x)
        x_norm = self.norm(x_proj)

        # 子空间划分
        subspaces = torch.split(x_norm, self.subspace_dims, dim=1)
        enhanced_subspaces = []

        # 动态 residual scale
        residual_scales = self.scale_mlp(x_norm)  # [B, num_subspaces]

        # 子空间增强
        for i, sub in enumerate(subspaces):
            enhanced = self.subspace_mlps[i](sub)
            enhanced_subspaces.append(enhanced * residual_scales[:, i].unsqueeze(-1))
        enhanced = torch.cat(enhanced_subspaces, dim=1)

        # 子空间交互
        inter_out = self.inter_subspace(enhanced)
        enhanced = enhanced + inter_out

        # 全局统计特征
        mean = x_proj.mean(dim=1, keepdim=True)
        std = x_proj.std(dim=1, keepdim=True)
        max_val = x_proj.max(dim=1, keepdim=True)[0]
        min_val = x_proj.min(dim=1, keepdim=True)[0]
        skew = ((x_proj - mean)**3).mean(dim=1, keepdim=True)
        kurt = ((x_proj - mean)**4).mean(dim=1, keepdim=True)
        global_stat = torch.cat([mean, std, max_val, min_val, skew, kurt], dim=1)  # [B, 6]

        # Gating
        gate_input = torch.cat([x_proj, global_stat], dim=1)  # [B, feature_dim + 6]
        gate = F.relu(self.gate_fc1(gate_input))
        gate = self.gate_fc2(gate)
        gate = torch.softmax(gate, dim=-1)  # [B, num_subspaces]

        # 子空间 gating
        gated_subspaces = []
        start = 0
        for i, d in enumerate(self.subspace_dims):
            end = start + d
            gated_subspaces.append(enhanced[:, start:end] * gate[:, i].unsqueeze(-1))
            start = end
        gated = torch.cat(gated_subspaces, dim=1)

        # 融合 + 跳跃连接
        fused = self.fusion(gated)
        out = x + fused
        out = self.skip_connection(out)
        out = self.norm(out)

        return (out, gate) if return_gating else out


def SSI(A: torch.Tensor, B: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    dot = torch.sum(A * B, dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    t_theta = t * theta
    coeff_a = torch.sin(theta - t_theta) / sin_theta
    coeff_b = torch.sin(t_theta) / sin_theta
    return coeff_a * A + coeff_b * B

logger = logging.getLogger(__name__)


# features.shape torch.Size([32, 1024])



@DETECTOR.register_module(module_name='effort')
class SFDSNet(nn.Module):
    def __init__(self, config=None):
        super(SFDSNet, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.feature_enhancer = DSFE(
            feature_dim=1024,               # 输入特征维度
            subspace_dims=[256, 256, 512],  # 子空间维度
            dropout=0.1,                    # Dropout比例
            reduction=16,                   # reduction参数，用于 gating 降维
            use_enhancer=True                # 是否启用增强模块
        )
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.use_SSI_random = True
        self.SSI_range = (0.2, 0.8)

        self.use_SSI_classwise = True
        self.SSI_classwise_range = (0.2, 0.8)
        self.token_freq_enhancer = FTA(
            embed_dim=1024,
            hidden_dim=256,
            spec_hidden=16,
            num_bands=4,
            dropout=0.1,
            max_token_residual=0.20,
            max_cls_residual=0.15,
            band_dropout=0.10,
            use_phase_gate=True,
        )
        
        
    def SSI_random_batch(self, features: torch.Tensor) -> torch.Tensor:
        """
        Randomly interpolate features within the batch (Mixup style)
        """
        if self.training and getattr(self.config, 'use_SSI_random', False):
            batch_size = features.size(0)
            idx = torch.randperm(batch_size).to(features.device)
            features_pair = features[idx]
            t = torch.rand(batch_size, 1).to(features.device)
            t0, t1 = getattr(self.config, 'SSI_range', (0.0, 1.0))
            t = t * (t1 - t0) + t0
            features = SSI(features, features_pair, t)
        return features

    # -------- 类内 SSI --------
    def SSI_classwise(self, batch: dict, features: torch.Tensor) -> torch.Tensor:
        """
        Interpolate features only within the same class
        """
        if self.training and getattr(self.config, 'use_SSI_classwise', False):
            labels = batch['label']
            for class_label in torch.unique(labels):
                class_mask = labels == class_label
                if class_mask.sum() < 2:
                    continue
                class_features = features[class_mask]
                num_embeddings = len(class_features)
                idx2 = torch.randperm(num_embeddings)
                A = class_features
                B = class_features[idx2]
                t = torch.rand((num_embeddings, 1), device=features.device, dtype=features.dtype)
                t0, t1 = getattr(self.config, 'SSI_classwise_range', (0.0, 1.0))
                t = t * (t1 - t0) + t0
                augmented = SSI(A, B, t)
                features[class_mask] = augmented.to(features.dtype)
        return features

    def build_backbone(self, config):

        clip_model = CLIPModel.from_pretrained("/bvg/Deepfake/preprocessing/clip-vit-large-patch14")  # the path of this folder in your disk (download from the above link)


        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024-1)


        return clip_model.vision_model

    def features(self, data_dict: dict):
        out = self.backbone(data_dict['image'])
        return {
            'pooler_output': out['pooler_output'],        # [B, 1024]
            'last_hidden_state': out['last_hidden_state'] # [B, N+1, 1024]
        }

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)



    def compute_weight_loss(self):
        weight_sum_dict = {}
        num_weight_dict = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                if str(weight_curr.size()) not in weight_sum_dict.keys():
                    weight_sum_dict[str(weight_curr.size())] = weight_curr
                    num_weight_dict[str(weight_curr.size())] = 1
                else:
                    weight_sum_dict[str(weight_curr.size())] += weight_curr
                    num_weight_dict[str(weight_curr.size())] += 1
        
        loss2 = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss2 += -torch.mean(S_sum)
        loss2 /= len(weight_sum_dict.keys())
        return loss2

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']  # Tensor of shape [batch_size]
        pred = pred_dict['cls']     # Tensor of shape [batch_size, num_classes]

        # Compute overall loss using all samples
        loss = self.loss_func(pred, label)

        # Create masks for real and fake classes
        mask_real = label == 0  # Boolean tensor
        mask_fake = label == 1  # Boolean tensor

        # Compute loss for real class
        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            # No real samples in batch
            loss_real = torch.tensor(0.0, device=pred.device)

        # Compute loss for fake class
        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            # No fake samples in batch
            loss_fake = torch.tensor(0.0, device=pred.device)
        


        loss_dict = {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
            # 'erank_loss': loss2
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict) 
            # 2. 随机 batch 内 SSI（可选）
        if self.training and getattr(self, 'use_SSI_random', False):
            features = self.SSI_random_batch(features)

        # 2. 类内 SSI（可选）
        if self.training and getattr(self, 'use_SSI_classwise', False):
            features = self.SSI_classwise(data_dict, features)
        feats = self.features(data_dict)

        pooler_output = feats['pooler_output']           # [B, 1024]
        last_hidden_state = feats['last_hidden_state']   # [B, N+1, 1024]

        cls_token = last_hidden_state[:, 0, :]           # [B, 1024]
        patch_tokens = last_hidden_state[:, 1:, :]       # [B, N, 1024]

        # token级频域增强
        freq_feat, freq_aux = self.token_freq_enhancer(
            cls_token, patch_tokens, return_aux=True
        )
        features = 0.5 * pooler_output + 0.5 * freq_feat
        features, gating_weights = self.feature_enhancer(features, return_gating=True)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            'cls': pred,
            'prob': prob,
            'feat': features,
            'stable_gate': gating_weights,
            'freq_band_weight': freq_aux['band_weights'] if freq_aux is not None else None,
            'freq_reg': freq_aux['freq_reg'] if freq_aux is not None else None,
            'token_alpha': freq_aux['token_alpha'] if freq_aux is not None else None,
            'cls_alpha': freq_aux['cls_alpha'] if freq_aux is not None else None,
        }

        return pred_dict


# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = torch.cat((self.U_r, self.U_residual), dim=1) @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            VVT = torch.cat((self.V_r, self.V_residual), dim=0) @ torch.cat((self.V_r, self.V_residual), dim=0).t()

            
            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
            
            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0
            
        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            # Total current weight is the fixed main weight plus the residual
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            
            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
            # loss = torch.abs(weight_current_fnorm ** 2 + 0.01 * self.weight_main_fnorm ** 2 - 1.01 * self.weight_original_fnorm ** 2)
        else:
            loss = 0.0
        
        return loss
    
    def compute_fn_loss(self):
        if (self.S_residual is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            
            loss = weight_current_fnorm ** 2
        else:
            loss = 0.0
        
        return loss

# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
            
            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module