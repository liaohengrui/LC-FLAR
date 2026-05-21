import torch
import torch.nn as nn



def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def maskedconv7x7_parallel(in_ch: int, out_ch: int, mask_type="5P") -> nn.Module:

    if mask_type not in ("5P", "4P", "3P", "2P", "P"):
        raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    maskedconv = MaskedConv2d("A", in_ch, out_ch, kernel_size=7, padding=0)

    maskedconv.mask[:, :, 2, 5:7] = 0

    return maskedconv

def maskedconv7x7_parallel_reverse(in_ch: int, out_ch: int, mask_type="5P") -> nn.Module:

    m = MaskedConv2dReverse("A",in_ch, out_ch, kernel_size=7, padding=0)

    m.mask[:, :, 4, 5:7] = 0

    return m


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type = "A", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1 :, :] = 0

    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

class MaskedConv2dReverse(nn.Conv2d):
    def __init__(self, mask_type = "A", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight))
        _, _, h, w = self.mask.size()
        cy, cx = h // 2, w // 2
        self.mask[:, :, cy, :cx] = 1
        self.mask[:, :, cy + 1:, :] = 1


    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)



