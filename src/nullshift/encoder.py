"""Frozen encoder wrapper: patch tokens only (CLS + registers dropped), optional L2 normalisation."""
from __future__ import annotations

from . import paths  # (sets cache env vars before timm import)
import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ENCODERS = {
    "dinov3_vitl16_sat": "vit_large_patch16_dinov3.sat493m",
    "dinov3_vitb16_web": "vit_base_patch16_dinov3.lvd1689m",   # 2nd backbone: smaller, web (not satellite) data
}


class Encoder(torch.nn.Module):
    def __init__(self, name: str = "dinov3_vitl16_sat", layer: int = -1, l2: bool = True, device: str = "cuda"):
        super().__init__()
        self.name, self.layer, self.l2 = name, layer, l2
        arch = ENCODERS[name]
        local = paths.MODELS / f"{arch}.fp16.safetensors"      # storage budget: fp16 copy (0.6 GB) instead of fp32 HF cache
        if local.exists():
            self.model = timm.create_model(arch.split(".")[0], pretrained=False, num_classes=0)
            self.model.load_state_dict({k: v.float() for k, v in load_file(str(local)).items()}, strict=True)
            self.model.pretrained_cfg = timm.models.get_pretrained_cfg(arch).to_dict()
            self.weights = str(local.name)
        else:
            self.model = timm.create_model(arch, pretrained=True)
            self.weights = "hf:" + arch
        self.model = self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device))
        self.patch = self.model.patch_embed.patch_size[0]
        self.dim = self.model.embed_dim
        self.n_blocks = len(self.model.blocks)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor, batch: int = 32) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1] -> (B,C,H/16,W/16) float32 features."""
        outs = []
        for i in range(0, x.shape[0], batch):
            xb = ((x[i:i + batch].to(self.device, torch.float32) - self.mean) / self.std)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                idx = [self.layer if self.layer >= 0 else self.n_blocks - 1]
                z = self.model.forward_intermediates(xb, indices=idx, norm=True, output_fmt="NCHW",
                                                     intermediates_only=True)[0]
            z = z.float()
            if self.l2:
                z = F.normalize(z, dim=1)
            outs.append(z)
        return torch.cat(outs)
