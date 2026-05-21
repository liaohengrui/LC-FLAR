import torch
import torchvision.transforms as T

from PIL import Image
import numpy as np


class PILToTensor(object):
    def __call__(self, pic):
        assert isinstance(pic, Image.Image), "PILToTensor: Please input a PIL image."

        img = torch.as_tensor(np.array(pic))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute((2,0,1)).float()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

def float_cdf_to_int16_normalized(cdf_float: torch.Tensor) -> torch.Tensor:
    """
    cdf_float: CPU tensor, float64/float32, shape (..., Lp), values in [0,1], nondecreasing, last ~ 1.
    return: CPU tensor, int16, shape (..., Lp), normalized for torchac.encode_int16_normalized_cdf
            NOTE: last element will be set to 0 to represent 2**16 on C++ side.
    """
    if cdf_float.device.type != "cpu":
        raise ValueError("cdf_float must be on CPU for deterministic int16 CDF building.")

    # Use float64 for stable rounding
    cdf = cdf_float.to(torch.float64)

    Lp = cdf.size(-1)
    TOT = 1 << 16  # 65536

    # Scale like torchac / numpyAc convention:
    # multiply by (2**16 - (Lp - 1)) then add arange(Lp) to guarantee strict increase after quantization
    scale = TOT - (Lp - 1)

    q = torch.round(cdf * scale).to(torch.int64)
    q = q + torch.arange(Lp, dtype=torch.int64).view(*([1] * (q.dim() - 1)), Lp)

    # Force endpoints
    q[..., 0] = 0
    q[..., -1] = TOT

    # Enforce strict monotonicity: q[i] >= q[i-1] + 1
    prev_plus_one = q[..., :-1] + 1
    q[..., 1:] = torch.maximum(q[..., 1:], prev_plus_one)

    # Also cap to TOT at the end
    q[..., -1] = TOT

    # torchac expects int16 on Python side, interpreted as uint16 in C++.
    # 2**16 cannot be represented in uint16, so final value must be 0 in int16 representation. :contentReference[oaicite:2]{index=2}
    q_int16 = q.to(torch.int32)
    q_int16[..., -1] = 0
    return q_int16.to(torch.int16)
def encode_convert_to_int_and_normalize(cdf_float, sym, check_input_bounds=False):
  if check_input_bounds:
    if cdf_float.min() < 0:
      raise ValueError(f'cdf_float.min() == {cdf_float.min()}, should be >=0.!')
    if cdf_float.max() > 1:
      raise ValueError(f'cdf_float.max() == {cdf_float.max()}, should be <=1.!')
    Lp = cdf_float.shape[-1]
    if sym.max() >= Lp - 1:
      raise ValueError
  cdf_int = _convert_to_int_and_normalize(cdf_float, True)
  return cdf_int

def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    PRECISION=16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(
        2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    return cdf

def build_transforms(transform_type):

    if transform_type == "p64":
        transform = T.Compose([
            T.RandomCrop(64),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            PILToTensor()
        ])
    elif transform_type == "p64_centercrop":
        transform = T.Compose([
            T.CenterCrop(64),
            PILToTensor()
        ])

    else:
        raise Exception("No existing transform type {}.".format(transform_type))

    return transform