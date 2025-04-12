import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from ultralytics.utils import yaml_load
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.utils.plotting import feature_visualization
from utils import build_simmim, build_yolohead, MaskGenerator
from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss


class YOLOwithSimMIM(BaseModel):
    def __init__(self, cfg=None, pretrain=False):
        super().__init__()
        self.yaml = cfg = cfg if isinstance(cfg, dict) else yaml_load(cfg)
        self.simmim = build_simmim(cfg)
        self.model, self.save = build_yolohead(cfg)
        self.mask_generator = MaskGenerator(
            cfg['img_size'],
            cfg['mask_patch_size'],
            cfg['patch_size'],
            cfg['mask_radio'])
        self.pretrain = pretrain

        self.inplace = cfg.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        _, width, max_channels = cfg["scales"][cfg.get("scale",'n')]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, make_divisible(min(cfg['embed_dim'],max_channels)*width,8)))
        trunc_normal_(self.mask_token, mean=0., std=.02)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = cfg['img_size']
            m.inplace = self.inplace

            def _forward(x):
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, 3, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        initialize_weights(self)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None, mask=None):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed, mask)

    def _predict_once(self, x, profile=False, visualize=False, embed=None, mask=None):
        y, dt, embeddings = [], [], []  # outputs

        x = self.simmim.encoder.patch_embed(x)

        if mask is not None:
            B, L, _ = x.shape
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - w) + mask_tokens * w

        if self.simmim.encoder.ape:
            x = x + self.simmim.encoder.absolute_pos_embed
        x = self.simmim.encoder.pos_drop(x)
        for layer in self.simmim.encoder.layers:
            x = layer(x)
            B,L,C = x.shape
            H = W = int(L**0.5)
            y.append(x.transpose(1,2).reshape(B,C,H,W))

        x = self.simmim.encoder.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)

        if self.pretrain:
            return self.simmim.decoder(x)

        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        if self.pretrain:
            batch_size = batch['img'].shape[0]
            mask = np.array([self.mask_generator() for _ in range(batch_size)])
            mask = torch.tensor(mask)
            preds = self.forward(batch['img'], mask=mask) if preds is None else preds
            mask = mask.repeat_interleave(self.simmim.patch_size, 1).repeat_interleave(self.simmim.patch_size, 2).unsqueeze(1).contiguous()
            if torch.cuda.is_available():
                mask = mask.to(torch.device('cuda:0'))
            loss_recon = F.l1_loss(batch['img'], preds, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.simmim.in_chans
            return loss, loss.detach()
        
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
