"""Running mean/std normalizer (MimicKit-compatible, single-process)."""

from __future__ import annotations

import numpy as np
import torch


class Normalizer(torch.nn.Module):
  def __init__(
    self,
    shape,
    device,
    init_mean=None,
    init_std=None,
    min_std=1e-4,
    clip=np.inf,
    dtype=torch.float,
    std_clip=None,
  ):
    super().__init__()
    self._min_var = min_std * min_std
    self._clip = clip
    self.dtype = dtype
    self.std_clip = std_clip
    self._build_params(shape, device, init_mean, init_std)

  def record(self, x: torch.Tensor) -> None:
    shape = self.get_shape()
    assert len(x.shape) > len(shape)
    x = torch.nan_to_num(x)
    x = x.flatten(start_dim=0, end_dim=len(x.shape) - len(shape) - 1)
    self._new_count += x.shape[0]
    self._new_sum += torch.sum(x, axis=0)
    self._new_sum_sq += torch.sum(torch.square(x), axis=0)

  def update(self) -> None:
    if self._mean_sq is None:
      self._mean_sq = self._calc_mean_sq(self._mean, self._std)

    new_count = self._new_count
    if new_count == 0:
      return
    new_mean = self._new_sum / new_count
    new_mean_sq = self._new_sum_sq / new_count

    new_total = self._count + new_count
    w_old = self._count.type(torch.float) / new_total.type(torch.float)
    w_new = float(new_count) / new_total.type(torch.float)

    self._mean[:] = w_old * self._mean + w_new * new_mean
    self._mean_sq[:] = w_old * self._mean_sq + w_new * new_mean_sq
    self._count[:] = new_total

    self._std[:] = self._calc_std(self._mean, self._mean_sq)
    if self.std_clip is not None:
      self._std.data.clamp_(min=self.std_clip)

    self._mean.data.copy_(torch.nan_to_num(self._mean.data))
    self._std.data.copy_(torch.nan_to_num(self._std.data))
    if self._mean_sq is not None:
      self._mean_sq.data.copy_(torch.nan_to_num(self._mean_sq.data))

    self._new_count = 0
    self._new_sum[:] = 0
    self._new_sum_sq[:] = 0

  def get_shape(self):
    return self._mean.shape

  def get_count(self):
    return self._count

  def get_mean(self):
    return self._mean

  def get_std(self):
    return self._std

  def set_mean_std(self, mean, std) -> None:
    shape = self.get_shape()
    assert mean.shape == shape and std.shape == shape
    self._mean[:] = mean
    self._std[:] = std
    self._mean_sq[:] = self._calc_mean_sq(self._mean, self._std)

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x)
    mean = torch.nan_to_num(self._mean)
    std = torch.nan_to_num(self._std)
    std = torch.clamp(
      std, min=torch.sqrt(torch.tensor(self._min_var, device=std.device, dtype=std.dtype))
    )
    norm_x = (x - mean) / std
    norm_x = torch.clamp(norm_x, -self._clip, self._clip)
    return torch.nan_to_num(norm_x).type(self.dtype)

  def unnormalize(self, norm_x: torch.Tensor) -> torch.Tensor:
    norm_x = torch.nan_to_num(norm_x)
    mean = torch.nan_to_num(self._mean)
    std = torch.nan_to_num(self._std)
    std = torch.clamp(
      std, min=torch.sqrt(torch.tensor(self._min_var, device=std.device, dtype=std.dtype))
    )
    x = norm_x * std + mean
    return torch.nan_to_num(x).type(self.dtype)

  def _calc_std(self, mean, mean_sq):
    var = mean_sq - torch.square(mean)
    var = torch.clamp_min(var, self._min_var)
    return torch.sqrt(var).type(self.dtype)

  def _calc_mean_sq(self, mean, std):
    return (torch.square(std) + torch.square(mean)).type(self.dtype)

  def _build_params(self, shape, device, init_mean, init_std) -> None:
    self._count = torch.nn.Parameter(
      torch.zeros([1], device=device, requires_grad=False, dtype=torch.long),
      requires_grad=False,
    )
    self._mean = torch.nn.Parameter(
      torch.zeros(shape, device=device, requires_grad=False, dtype=self.dtype),
      requires_grad=False,
    )
    self._std = torch.nn.Parameter(
      torch.ones(shape, device=device, requires_grad=False, dtype=self.dtype),
      requires_grad=False,
    )
    self._mean_sq = None
    if init_mean is not None:
      self._mean[:] = init_mean
    if init_std is not None:
      self._std[:] = init_std
    self._new_count = 0
    self._new_sum = torch.zeros_like(self._mean)
    self._new_sum_sq = torch.zeros_like(self._mean)
