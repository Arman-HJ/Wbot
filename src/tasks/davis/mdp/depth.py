"""Depth alignment and domain randomization (DAVIS Appendix F / Fig. 4)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class DepthPreprocessCfg:
  """Paper-aligned depth pipeline; camera-specific crop is configurable."""

  # Paper: clip 0.3–5.0 m, normalize 2D/5−1, invalid → 0.
  depth_min_m: float = 0.3
  depth_max_m: float = 5.0
  render_height: int = 94
  render_width: int = 168
  crop_height: int = 80
  crop_width: int = 168
  # Domain randomization (Table XIII). Probabilities are paper values.
  noise_distance_coef: float = 0.01
  noise_gaussian_coef: float = 0.005
  salt_pepper_prob: float = 0.01
  stripe_dropout_prob: float = 0.005
  edge_drag_prob: float = 0.01
  rect_mask_prob: float = 0.1
  rect_mask_size_frac: tuple[float, float] = (0.10, 0.25)
  ball_occlusion_prob: float = 0.2
  ball_occlusion_radius_px: tuple[int, int] = (2, 10)
  # Project page App. sim2real: partial occlusion 45–85% when applied.
  partial_occlusion_prob: float = 0.05
  partial_occlusion_frac: tuple[float, float] = (0.45, 0.85)
  border_nearfield_prob: float = 0.15
  # Edge drag: shift a strip of border pixels inward (approx of rolling-shutter / blur).
  edge_drag_prob: float = 0.01
  edge_drag_px: int = 2


def crop_center(depth: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
  """Center-crop (B,1,H,W) to (B,1,out_h,out_w)."""
  _, _, h, w = depth.shape
  if h == out_h and w == out_w:
    return depth
  top = max((h - out_h) // 2, 0)
  left = max((w - out_w) // 2, 0)
  return depth[:, :, top : top + out_h, left : left + out_w]


def normalize_depth(
  depth_m: torch.Tensor,
  depth_min_m: float,
  depth_max_m: float,
) -> torch.Tensor:
  """Clip, normalize as 2D/max−1, zero invalid / out-of-range (paper App. F)."""
  valid = torch.isfinite(depth_m) & (depth_m > 0)
  clipped = depth_m.clamp(depth_min_m, depth_max_m)
  # Paper: 2D/5.0 − 1  → range [-1, 1] for valid depths in [0, 5].
  norm = 2.0 * clipped / depth_max_m - 1.0
  out = torch.where(valid & (depth_m >= depth_min_m) & (depth_m <= depth_max_m), norm, torch.zeros_like(norm))
  # Values below min or invalid → 0 (paper: zero-filled at invalid pixels).
  out = torch.where(valid & (depth_m >= depth_min_m), out, torch.zeros_like(out))
  return out


def randomize_depth(
  depth_norm: torch.Tensor,
  cfg: DepthPreprocessCfg,
  *,
  training: bool = True,
) -> torch.Tensor:
  """Apply Table XIII-style perturbations on normalized depth (training only)."""
  if not training:
    return depth_norm
  x = depth_norm
  b, _, h, w = x.shape
  device = x.device

  # Distance-proportional + Gaussian noise on the metric-equivalent scale.
  # Operate in normalized space with paper coefficients as relative scales.
  # 论文把系数定义在度量深度上；此处在归一化图上做等价扰动（标注为合理默认映射）。
  noise = cfg.noise_gaussian_coef * torch.randn_like(x)
  noise = noise + cfg.noise_distance_coef * torch.randn_like(x) * (x.abs() + 1e-3)
  x = (x + noise).clamp(-1.0, 1.0)

  if cfg.salt_pepper_prob > 0:
    sp = torch.rand(b, 1, h, w, device=device)
    x = torch.where(sp < cfg.salt_pepper_prob * 0.5, torch.zeros_like(x), x)
    x = torch.where(sp > 1.0 - cfg.salt_pepper_prob * 0.5, torch.ones_like(x) * -1.0, x)

  if cfg.stripe_dropout_prob > 0:
    # Random vertical/horizontal stripes set to 0.
    row_mask = torch.rand(b, 1, h, 1, device=device) < cfg.stripe_dropout_prob
    col_mask = torch.rand(b, 1, 1, w, device=device) < cfg.stripe_dropout_prob
    x = x.masked_fill(row_mask | col_mask, 0.0)

  if cfg.rect_mask_prob > 0:
    apply = torch.rand(b, device=device) < cfg.rect_mask_prob
    if apply.any():
      for i in apply.nonzero(as_tuple=False).flatten().tolist():
        frac = torch.empty(1, device=device).uniform_(*cfg.rect_mask_size_frac).item()
        rh = max(int(h * frac), 1)
        rw = max(int(w * frac), 1)
        y0 = int(torch.randint(0, max(h - rh + 1, 1), (1,), device=device).item())
        x0 = int(torch.randint(0, max(w - rw + 1, 1), (1,), device=device).item())
        x[i, :, y0 : y0 + rh, x0 : x0 + rw] = 0.0

  if cfg.ball_occlusion_prob > 0:
    apply = torch.rand(b, device=device) < cfg.ball_occlusion_prob
    if apply.any():
      r_lo, r_hi = cfg.ball_occlusion_radius_px
      for i in apply.nonzero(as_tuple=False).flatten().tolist():
        rad = int(torch.randint(r_lo, r_hi + 1, (1,), device=device).item())
        cy = int(torch.randint(0, h, (1,), device=device).item())
        cx = int(torch.randint(0, w, (1,), device=device).item())
        yy = torch.arange(h, device=device).view(-1, 1)
        xx = torch.arange(w, device=device).view(1, -1)
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
        x[i, 0] = torch.where(disk, torch.zeros_like(x[i, 0]), x[i, 0])

  if cfg.border_nearfield_prob > 0:
    apply = torch.rand(b, device=device) < cfg.border_nearfield_prob
    if apply.any():
      border = max(h // 16, 1)
      x[apply, :, :border, :] = 0.0
      x[apply, :, -border:, :] = 0.0
      x[apply, :, :, :border] = 0.0
      x[apply, :, :, -border:] = 0.0

  if getattr(cfg, "partial_occlusion_prob", 0.0) > 0:
    apply = torch.rand(b, device=device) < cfg.partial_occlusion_prob
    if apply.any():
      for i in apply.nonzero(as_tuple=False).flatten().tolist():
        frac = torch.empty(1, device=device).uniform_(*cfg.partial_occlusion_frac).item()
        rh = max(int(h * math.sqrt(frac)), 1)
        rw = max(int(w * math.sqrt(frac)), 1)
        y0 = int(torch.randint(0, max(h - rh + 1, 1), (1,), device=device).item())
        x0 = int(torch.randint(0, max(w - rw + 1, 1), (1,), device=device).item())
        x[i, :, y0 : y0 + rh, x0 : x0 + rw] = 0.0

  if getattr(cfg, "edge_drag_prob", 0.0) > 0:
    apply = torch.rand(b, device=device) < cfg.edge_drag_prob
    if apply.any():
      px = max(int(cfg.edge_drag_px), 1)
      # Copy near-border columns/rows inward (cheap edge-drag approx).
      x[apply, :, :, px : 2 * px] = x[apply, :, :, :px]
      x[apply, :, px : 2 * px, :] = x[apply, :, :px, :]

  return x


def align_depth(
  depth_m: torch.Tensor,
  cfg: DepthPreprocessCfg,
  *,
  training: bool,
) -> torch.Tensor:
  """Full sim/real shared path: crop → normalize → optional DR.

  Input ``depth_m``: (B,1,H,W) meters from the renderer.
  Output: (B,1,crop_h,crop_w) normalized.
  """
  depth_m = crop_center(depth_m, cfg.crop_height, cfg.crop_width)
  depth_n = normalize_depth(depth_m, cfg.depth_min_m, cfg.depth_max_m)
  return randomize_depth(depth_n, cfg, training=training)
