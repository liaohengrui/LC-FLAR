import argparse
import math

import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchac

from flar_model_eval import LosslessCompressor
import pickle
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def coding_order_table7x7(patch_sz=64):

    COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)

    for i in range(32, patch_sz):
        start = 2 * (i - 32) + 1
        COT[i, :] = torch.arange(start, start + patch_sz)
    for j in range(32):
        start = 9 + j * 2
        COT[31 - j, :] = torch.arange(start, start + patch_sz)


    return COT


def img2patch(img, patch_sz):
    h, w, _ = img.shape
    h_num = h // patch_sz - (patch_sz - (h % patch_sz)) // patch_sz + 1
    w_num = w // patch_sz - (patch_sz - (w % patch_sz)) // patch_sz + 1

    img_pad = nn.functional.pad(img,
                                (0, 0, (patch_sz - w % patch_sz) % patch_sz, 0, (patch_sz - h % patch_sz) % patch_sz,
                                 0))

    patch_h = torch.chunk(img_pad, h_num, dim=0)
    patch_h = torch.stack(patch_h, dim=0)

    patch_w = torch.chunk(patch_h, w_num, dim=2)
    patch = torch.cat(patch_w, dim=0)

    patch = patch.permute(0, 3, 1, 2)

    return patch


def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image).astype(np.float32)
    return image


def compress(model, input, COT, patch_sz=64, mix_num=5):
    norm_scale = 1 / 255.
    half = 0.5 * norm_scale
    mix_num2 = 2 * mix_num
    code_res = []
    img_shape = input.shape[:2]
    bin_sz = 1

    device = next(model.parameters()).device

    x = img2patch(torch.as_tensor(input), patch_sz=patch_sz).to(device)
    x = x.to(dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)

    high = math.ceil(img_shape[0] / 64)
    width = math.ceil(img_shape[1] / 64)
    cur_index = -1
    patch_2 = []
    if high % 2 == 0:
        flag = True
        for i in range((high // 2 * width)):
            step = 2
            if i % (high // 2) == 0 and flag and i != 0:
                step = 1
                flag = False
            elif i % (high // 2) == 0 and not flag and i != 0:
                step = 3
                flag = True
            cur_index += step
            patch_2.append(cur_index)
    else:
        while cur_index + 2 < high * width:
            cur_index += 2
            patch_2.append(cur_index)

    B = x.shape[0]
    mask = torch.ones(B, device=device, dtype=torch.bool)
    idx2 = torch.as_tensor(patch_2, device=device, dtype=torch.long)
    mask[idx2] = False
    idx1 = torch.arange(B, device=device, dtype=torch.long)[mask]
    x1 = x.index_select(0, idx1)
    x2 = x.index_select(0, idx2)

    x2_pad = torch.zeros((x2.shape[0], 3, 64 + 2 * 3, 64 + 2 * 3),
                         device=device).to(dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
    for i in range(len(patch_2)):
        original_idx = patch_2[i]
        original_idx_up = original_idx - 1
        original_idx_down = original_idx + 1
        original_idx_left = original_idx - high
        original_idx_right = original_idx + high

        x2_pad[i, :, 3:3 + 64, 3:3 + 64] = x[original_idx]
        # if original_idx % high != 0:
        #     x2_pad[i, :, 0:3, 3:3 + 64] = x[original_idx_up, :, -3:, :]
        # if original_idx % (high - 1) != 0:
        #     x2_pad[i, :, 3 + 64:3 + 64 + 3, 3:3 + 64] = x[original_idx_down, :, -3:, :]
        if original_idx_left >= 0:
            x2_pad[i, :, 3:3 + 64, 0:3] = x[original_idx_left, :, :, -3:]
        if original_idx_right < high * width:
            x2_pad[i, :, 3:3 + 64, 3 + 64:3 + 64 + 3] = x[original_idx_right, :, :, :3]

    with torch.no_grad():
        for j in range(2):
            if j == 0:
                x_f = F.pad(x1, pad=(3, 3, 3, 3), mode="constant", value=0)
                x = x1
            else:
                x_f = x2_pad
                x = x2

            samples = torch.arange(0, 255 + 1, step=bin_sz, dtype=torch.float32).to(device)
            samples = samples * norm_scale
            res_q_max_norm = 255 * norm_scale
            res_q_min_norm = 0 * norm_scale

            res_tmp = torch.zeros_like(x)

            # ctx_total = model.mask_conv(x_f * norm_scale)
            x_f_1 = x_f[:, :, :35, :]  # (B,C,32,64)
            x_f_2 = x_f[:, :, 35:, :]  # (B,C,32,64)
            x_f_1 = F.pad(x_f_1, (0, 0, 0, 3), mode="constant", value=0)
            x_f_1[:, :, -3:, :] = x_f_2[:, :, :3, :]
            x_f_2 = F.pad(x_f_2, (0, 0, 3, 0), mode="constant", value=0)
            ctx1 = model.mask_conv_reverse(x_f_1 * norm_scale)
            ctx2 = model.mask_conv(x_f_2 * norm_scale)
            ctx_total = torch.cat([ctx1, ctx2], dim=2)

            max_step = torch.max(COT)

            for i in range(max_step):
                h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
                ctx = ctx_total[:, :, h_idx, w_idx].unsqueeze(3)

                res_crop = x[:, :, h_idx, w_idx]
                res_tmp_crop = res_tmp[:, :, h_idx, w_idx]

                res_tmp_crop = res_tmp_crop.unsqueeze(3)
                lmm_params = model.residual_compressor(ctx)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
                coeffs = torch.tanh(coeffs)

                for c in range(3):
                    if c == 0:
                        mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)
                    elif c == 1:
                        mu_c = mu[:, mix_num:mix_num2, :, :] + (res_tmp_crop[:, 0:1, :, :] * norm_scale) * coeffs[:,
                                                                                                           :mix_num, :,
                                                                                                           :]
                        mu_c = mu_c.permute(0, 2, 1, 3)
                    else:
                        mu_c = mu[:, mix_num2:, :, :] + (res_tmp_crop[:, 0:1, :, :] * norm_scale) * coeffs[:,
                                                                                                    mix_num:mix_num2, :,
                                                                                                    :] + \
                               (res_tmp_crop[:, 1:2, :, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                        mu_c = mu_c.permute(0, 2, 1, 3)

                    samples_centered = samples - mu_c

                    inv_sigma = torch.exp(
                        -log_sigma[:, c * mix_num:(c + 1) * mix_num, :, :]
                        .permute(0, 2, 1, 3)
                    )

                    plus_in = inv_sigma * (samples_centered + half)
                    min_in = inv_sigma * (samples_centered - half)

                    cdf_plus = 0.5 * (1.0 + torch.erf(plus_in / math.sqrt(2.0)))
                    cdf_min = 0.5 * (1.0 + torch.erf(min_in / math.sqrt(2.0)))

                    cdf_delta = cdf_plus - cdf_min

                    # boundary handling
                    one_minus_cdf_min = torch.clamp(1.0 - cdf_min, min=1e-12)
                    cdf_plus = torch.clamp(cdf_plus, min=1e-12)
                    samples2 = samples - torch.zeros_like(mu_c)

                    cdf_delta = torch.where(
                        samples2 - half < res_q_min_norm,
                        cdf_plus,
                        torch.where(
                            samples2 + half > res_q_max_norm,
                            one_minus_cdf_min,
                            cdf_delta
                        )
                    )

                    weights_c = weights.permute(0, 2, 1, 3)
                    m = torch.amax(weights_c, 2, keepdim=True)
                    weights_c = torch.exp(
                        weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                    pmf = torch.sum(cdf_delta * weights_c, dim=2)

                    pmf = pmf.clamp_(1. / 64800, 1.)
                    pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                    cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                    cdf = F.pad(cdf, (1, 0))

                    symbol = torch.div(res_crop[:, c, :].short() - 0, bin_sz, rounding_mode='floor')

                    res_stream = torchac.encode_float_cdf(cdf.cpu(), symbol.cpu(), needs_normalization=False,
                                                          check_input_bounds=False)
                    code_res.append(res_stream)
                    res_tmp_crop[:, c, :, 0] = symbol.float() * bin_sz + 0

    return code_res, img_shape


if __name__ == '__main__':
    COT = coding_order_table7x7()

    ckp_dir = "./ckp_ll"
    I = load_image('./img/wedding.png')

    device = torch.device('cuda')
    ll_module = LosslessCompressor(192).eval().to(device)

    ckp = torch.load(os.path.join(ckp_dir, "ckp_best.tar"), map_location=device)
    ll_module.load_state_dict(ckp['model_state_dict'])
    start = time.perf_counter()
    code_res, img_shape = compress(ll_module, I, COT)

    res_sz = sum([len(code_res[i]) for i in range(len(code_res))])

    bpsp = res_sz * 8 / np.prod(I.shape)

    print("bpsp:{:.4f}".format(bpsp))

    with open('./img/Bitstream.bin',
              'wb') as f:
        pickle.dump((code_res, img_shape), f)

    print(f"Compression finished. Bitstream saved to ./img/Bitstream.bin ")
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} s")
