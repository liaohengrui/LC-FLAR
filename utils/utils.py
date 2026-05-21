import torch
import torch.nn.functional as F


def split_and_pad(x: torch.Tensor, pad: int = 3):
    """
    x: (B, C, 64, 64)

    returns:
      x1_pad: (B, C, 32+2*pad, 64+2*pad) with bottom pad rows copied from x2 top rows
      x2_pad: (B, C, 32+2*pad, 64+2*pad) zero padded
    """
    B, C, H, W = x.shape
    assert H == 64 and W == 64, f"Expected (B,C,64,64), got {x.shape}"
    assert pad == 3, "This implementation assumes pad=3 per your description."

    # split
    x1 = x[:, :, :32, :]   # (B,C,32,64)
    x2 = x[:, :, 32:, :]   # (B,C,32,64)

    # x2: zero pad 3 on all sides
    x2_pad = F.pad(x2, (pad, pad, pad, pad), mode="constant", value=0)

    # x1: start with zero pad 3 on all sides
    x1_pad = F.pad(x1, (pad, pad, pad, pad), mode="constant", value=0)

    # replace x1's bottom padding 3 rows with x2's top 3 rows (with left/right padding zeros)
    # x2_top: (B,C,3,64) -> pad left/right to (B,C,3,70)
    x2_top = x2[:, :, :pad, :]  # top 3 rows of x2
    x2_top_lr = F.pad(x2_top, (pad, pad, 0, 0), mode="constant", value=0)  # (B,C,3,70)

    # x1_pad bottom 3 rows are indices [-pad:, :] in height dimension
    x1_pad[:, :, -pad:, :] = x2_top_lr

    return x1_pad, x2_pad