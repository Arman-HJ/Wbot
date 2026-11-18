"""LightDepthEncoder: depth (B,1,80,168) → 32-D latent.

论文只给出输出维 32 与名称，未公开卷积层配置。下列为最小合理默认，标注假设。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LightDepthEncoder(nn.Module):
  """论文未给出层结构；暂用 3×Conv + AdaptiveAvgPool + Linear→32。"""

  def __init__(
    self,
    in_channels: int = 1,
    latent_dim: int = 32,
    channels: tuple[int, ...] = (16, 32, 32),
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    c_in = in_channels
    for c_out in channels:
      layers.extend(
        [
          nn.Conv2d(c_in, c_out, kernel_size=5 if c_in == in_channels else 3, stride=2, padding=2),
          nn.ELU(inplace=True),
        ]
      )
      c_in = c_out
    self.backbone = nn.Sequential(*layers)
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(c_in, latent_dim)

  def forward(self, depth: torch.Tensor) -> torch.Tensor:
    """Args:
      depth: (B, 1, H, W) normalized depth.
    Returns:
      (B, 32) latent.
    """
    x = self.backbone(depth)
    x = self.pool(x).flatten(1)
    return self.fc(x)
