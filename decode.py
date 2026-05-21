import math
import time

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchac

from flar_model_eval import LosslessCompressor
import os
import pickle

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def coding_order_table7x7(patch_sz=64, mask_type="3P"):

    COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)

    for i in range(32, patch_sz):
        start = 2 * (i - 32) + 1
        COT[i, :] = torch.arange(start, start + patch_sz)
    for j in range(32):
        start = 9 + j * 2
        COT[31 - j, :] = torch.arange(start, start + patch_sz)

    return COT


def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image).astype(np.float32)
    return image


def patch2img(patch, img_sz):
    h, w = img_sz
    patch_sz = patch.shape[2]
    h_num = h // patch_sz - (patch_sz - (h % patch_sz)) // patch_sz + 1
    w_num = w // patch_sz - (patch_sz - (w % patch_sz)) // patch_sz + 1

    patch = patch.permute(0, 2, 3, 1)
    patch_w = torch.chunk(patch, w_num, dim=0)
    patch_h = torch.cat(patch_w, dim=2)

    patch_h = torch.chunk(patch_h, h_num, dim=0)
    img = torch.cat(patch_h, dim=1).squeeze(0)

    return img[-h:, -w:, :]


def decompress(model, code_res, img_shape, COT, mix_num=5):
    device = next(model.parameters()).device

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

    B = high*width
    mask = torch.ones(B, device=device, dtype=torch.bool)
    idx2 = torch.as_tensor(patch_2, device=device, dtype=torch.long)
    mask[idx2] = False
    idx1 = torch.arange(B, device=device, dtype=torch.long)[mask]


    norm_scale = 1 / 255.
    half = 0.5 * norm_scale
    mix_num2 = 2 * mix_num
    bin_sz = 1


    res_q_min, res_q_max = 0, 255
    res_q_max_norm = res_q_max * norm_scale
    res_q_min_norm = res_q_min * norm_scale


    samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
    samples = samples * norm_scale
    w, h = img_shape
    patch_sz = 64
    h_num = (h + patch_sz - 1) // patch_sz
    w_num = (w + patch_sz - 1) // patch_sz
    dtype = torch.float32

    res_tmp_total = torch.zeros((h_num * w_num, 3, 64, 64), device=device, dtype=dtype) \
        .contiguous(memory_format=torch.contiguous_format)

    with torch.no_grad():
        for k in range(2):
            if k == 0:
                num_blocks = idx1.shape[0]
                j = 0
            else:
                num_blocks = idx2.shape[0]
                i = 0

            res_tmp = torch.zeros((num_blocks, 3, 64, 64), device=device, dtype=dtype) \
                .contiguous(memory_format=torch.contiguous_format)
            res_tmp_f = torch.zeros((num_blocks, 3, 70, 70), device=device, dtype=dtype) \
                .contiguous(memory_format=torch.contiguous_format)
            if k == 1:
                for l in range(num_blocks):
                    original_idx = idx2[l].item()
                    original_idx_up = original_idx - 1
                    original_idx_left = original_idx - high
                    original_idx_right = original_idx + high

                    res_tmp_f[l, :, 3:3 + 64, 3:3 + 64] = res_tmp_total[original_idx]
                    if original_idx % high != 0:
                        res_tmp_f[l, :, 0:3, 3:3 + 64] = res_tmp_total[original_idx_up, :, -3:, :]
                    if original_idx_left >= 0:
                        res_tmp_f[l, :, 3:3 + 64, 0:3] = res_tmp_total[original_idx_left, :, :, -3:]
                    if original_idx_right < high * width:
                        res_tmp_f[l, :, 3:3 + 64, 3 + 64:3 + 64 + 3] = res_tmp_total[original_idx_right, :, :, :3]


            max_step = torch.max(COT)

            for i in range(max_step):

                h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
                res_tmp_f_1 = res_tmp_f[:, :, :35, :]
                res_tmp_f_2 = res_tmp_f[:, :, 35:, :]
                res_tmp_f_1 = F.pad(res_tmp_f_1, (0, 0, 0, 3), mode="constant", value=0)
                res_tmp_f_1[:, :, -3:, :] = res_tmp_f_2[:, :, :3, :]
                res_tmp_f_2 = F.pad(res_tmp_f_2, (0, 0, 3, 0), mode="constant", value=0)
                # --------------------replace optimaze gpu memory---------------------
                # ctx1 = model.mask_conv_reverse(res_tmp_f_1 * norm_scale)
                # ctx2 = model.mask_conv(res_tmp_f_2 * norm_scale)
                # ctx_total = torch.cat([ctx1, ctx2], dim=2)
                #
                # # ctx = model.mask_conv(res_tmp_f * norm_scale)[:, :, h_idx, w_idx].unsqueeze(3)
                # --------------------replace optimaze gpu memory---------------------

                # --------------------replace optimaze gpu memory---------------------
                ctx1 = model.mask_conv_reverse(res_tmp_f_1 * norm_scale)  # shape: [B, C, H1, W]
                ctx2 = model.mask_conv(res_tmp_f_2 * norm_scale)  # shape: [B, C, H2, W]

                B, C, H1, W = ctx1.shape
                H2 = ctx2.shape[2]

                ctx_total = ctx1.new_empty((B, C, H1 + H2, W))
                ctx_total[:, :, :H1, :] = ctx1
                ctx_total[:, :, H1:, :] = ctx2
                # --------------------replace optimaze gpu memory---------------------

                ctx = ctx_total[:, :, h_idx, w_idx].unsqueeze(3)

                del ctx_total
                torch.cuda.empty_cache()
                res_crop = res_tmp[:, :, h_idx, w_idx].unsqueeze(3)

                lmm_params = model.residual_compressor(ctx)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
                coeffs = torch.tanh(coeffs)

                for c in range(3):
                    if c == 0:
                        mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)
                    elif c == 1:
                        mu_c = mu[:, mix_num:mix_num2, :, :] + (res_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, :mix_num,
                                                                                                       :, :]
                        mu_c = mu_c.permute(0, 2, 1, 3)
                    else:
                        mu_c = mu[:, mix_num2:, :, :] + (res_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, mix_num:mix_num2,
                                                                                                :, :] + \
                               (res_crop[:, 1:2, :, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                        mu_c = mu_c.permute(0, 2, 1, 3)

                    samples_centered = samples - mu_c

                    inv_sigma = torch.exp(
                        -log_sigma[:, c * mix_num:(c + 1) * mix_num, :, :]
                        .permute(0, 2, 1, 3)
                    )

                    half = 0.5 / 255.0

                    # -------- Gaussian CDF --------
                    plus_in = inv_sigma * (samples_centered + half)
                    min_in = inv_sigma * (samples_centered - half)

                    cdf_plus = 0.5 * (1.0 + torch.erf(plus_in / math.sqrt(2.0)))
                    cdf_min = 0.5 * (1.0 + torch.erf(min_in / math.sqrt(2.0)))

                    cdf_delta = cdf_plus - cdf_min

                    # boundary handling (MUST match encoder)
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

                    # -------- mixture weights --------
                    weights_c = weights.permute(0, 2, 1, 3)
                    m = torch.amax(weights_c, 2, keepdim=True)
                    weights_c = torch.exp(
                        weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True))
                    )

                    # -------- PMF / CDF --------
                    pmf = torch.sum(cdf_delta * weights_c, dim=2)

                    pmf = pmf.clamp_(1. / 64800, 1.)
                    pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)

                    cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                    cdf = F.pad(cdf, (1, 0))

                    # -------- arithmetic decode --------
                    symbol_out = torchac.decode_float_cdf(
                        cdf.cpu(),
                        code_res[j],
                        needs_normalization=False
                    )
                    res_crop[:, c, :, 0] = symbol_out.float() * bin_sz + res_q_min
                    j += 1
                res_tmp[:, :, h_idx, w_idx] = res_crop.squeeze(3)
                res_tmp_f[:, :, 3+h_idx, 3+w_idx] = res_crop.squeeze(3)
            if k == 0:
                res_tmp_total[idx1] = res_tmp
            else:
                res_tmp_total[idx2] = res_tmp





    res = patch2img(res_tmp_total, img_shape)

    x_rec = (res).clamp_(min=0, max=255)

    return x_rec


if __name__ == '__main__':
    ckp_dir = "./ckp_ll"

    device = torch.device('cuda')
    ll_module = LosslessCompressor(192).eval().to(device)

    ckp = torch.load(os.path.join(ckp_dir, "ckp_best.tar"), map_location=device)
    ll_module.load_state_dict(ckp['model_state_dict'])

    COT = coding_order_table7x7()
    start = time.perf_counter()
    with open('./img/Bitstream.bin', 'rb') as f:
        code_res, img_shape = pickle.load(f)

    I_ll = decompress(ll_module, code_res, img_shape, COT)
    I_ll = I_ll.cpu().numpy()

    I_ll = I_ll.astype(np.uint8)
    im_ll = Image.fromarray(I_ll)
    im_ll.save('./img/rec.png')

    print(f"Decompression finished. Reconstructed image saved to ./img/rec.png")
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} s")
