import torch
import torch.nn as nn


from custom_layers import conv1x1, maskedconv7x7_parallel, \
    maskedconv7x7_parallel_reverse


from gaussmixturemodel import DiscretizedGaussianMixtureModel
from utils.utils import split_and_pad



class ResBlock_1x1(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        out = self.layers(input)
        out = out + input

        return out


class PiexCompressor(nn.Module):
    def __init__(self, num_ch, num_mixtures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ResBlock = ResBlock_1x1(num_ch)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, 10*num_mixtures),
        )
         
    def forward(self, input):
        out = self.ResBlock(input)
        out = self.layers(out)
        return out
    
        
class LosslessCompressor(nn.Module):
    def __init__(self, num_ch, mask_type="3P", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_conv = maskedconv7x7_parallel(3, 256, mask_type)
        self.mask_conv_reverse = maskedconv7x7_parallel_reverse(3, 256, mask_type)
        self.residual_compressor = PiexCompressor(256, 5)

    def forward(self, input):
        x = input / 255.
        # context = self.mask_conv(x)
        # Equivalent context reordering strategy
        x1,x2 = split_and_pad(x)
        ctx1 = self.mask_conv_reverse(x1)
        ctx2 = self.mask_conv(x2)
        context = torch.cat([ctx1, ctx2], dim=2)

        lmm_params = self.piex_compressor(context)
        mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
        lmm = DiscretizedGaussianMixtureModel(mu, log_sigma, weights, coeffs)
        log_res_likelihoods = lmm(x)

        return {
            "res_likelihoods": log_res_likelihoods,
        }

