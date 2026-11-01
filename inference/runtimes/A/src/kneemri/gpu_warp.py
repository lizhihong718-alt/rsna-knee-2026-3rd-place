"""cv2.warpAffine(uint8, M, INTER_LANCZOS4, BORDER_CONSTANT 0) re-implemented on the GPU, bit-exact.

The native384 recipe warps every training window with a Lanczos-4 rotation/scale (``warp_interpolation:
lanczos4``). At 384 that costs ~10 ms per frame on a throttled CPU core (~0.9 core-s per study), which
starves the GPU. This module reproduces OpenCV's integer pipeline so that the training pixels stay
identical while the loader only draws the random angle/scale:

  * M is inverted in double exactly as cv::warpAffine does (no WARP_INVERSE_MAP);
  * source coordinates use AB_BITS=10 fixed point, rounded half-to-even (cvRound), + AB_SCALE/64,
    then quantised to INTER_TAB_SIZE=32 sub-pixel positions;
  * the 8x8 weight for each of the 32x32 sub-pixel positions is the product of OpenCV's float32
    interpolateLanczos4 coefficients, scaled to Q15 (INTER_REMAP_COEF_SCALE=32768) with OpenCV's
    sum-correction on the central 2x2 taps;
  * out = saturate_uchar((sum_k w_k * src_k + 2**14) >> 15); taps outside the image read 0.

``check_against_cv2`` (and analysis/check_gpu_warp.py) compares against cv2 on real windows.
"""
import math

import numpy as np
import torch

AB_BITS, INTER_BITS, TAB = 10, 5, 32
AB_SCALE = 1 << AB_BITS
ROUND_DELTA = AB_SCALE // TAB // 2
COEF_BITS = 15
_ITAB = {}


def _lanczos4_1d(x):
    """OpenCV interpolateLanczos4 with its float32/double mix; x is a float32 sub-pixel offset."""
    s45 = 0.70710678118654752440084436210485
    cs = [(1, 0), (-s45, -s45), (0, 1), (s45, -s45), (-1, 0), (s45, s45), (0, -1), (-s45, s45)]
    x = np.float32(x)
    y0 = float(-(x + np.float32(3))) * math.pi * 0.25
    s0, c0 = math.sin(y0), math.cos(y0)
    coeffs = np.zeros(8, np.float32)
    total = np.float32(0)
    for i in range(8):
        y0_ = np.float32(x + np.float32(3) - np.float32(i))
        if abs(y0_) >= np.float32(1e-6):
            y = float(-y0_) * math.pi * 0.25
            coeffs[i] = np.float32((cs[i][0] * s0 + cs[i][1] * c0) / (y * y))
        else:
            coeffs[i] = np.float32(1e30)
        total = np.float32(total + coeffs[i])
    inv = np.float32(np.float32(1) / total)
    return (coeffs * inv).astype(np.float32)


def lanczos4_table():
    """int32 [TAB*TAB, 64]: row = (fy*TAB + fx), column = ky*8 + kx (OpenCV initInterTab2D, fixed point)."""
    tab1 = np.stack([_lanczos4_1d(np.float32(i) * np.float32(1.0 / TAB)) for i in range(TAB)])
    out = np.zeros((TAB * TAB, 64), np.int32)
    for i in range(TAB):
        for j in range(TAB):
            v = (tab1[i][:, None] * tab1[j][None, :]).astype(np.float32)
            itab = np.rint(v * np.float32(1 << COEF_BITS)).astype(np.int64).reshape(8, 8)
            diff = int(itab.sum()) - (1 << COEF_BITS)
            if diff:
                mk = Mk = (4, 4)
                for k1 in (4, 5):
                    for k2 in (4, 5):
                        if itab[k1, k2] < itab[mk]:
                            mk = (k1, k2)
                        elif itab[k1, k2] > itab[Mk]:
                            Mk = (k1, k2)
                if diff < 0:
                    itab[Mk] -= diff
                else:
                    itab[mk] -= diff
            out[i * TAB + j] = itab.reshape(-1)
    return out


def invert_affine(m):
    """cv::warpAffine's in-place inversion of a forward 2x3 matrix (double arithmetic, same order)."""
    m = [float(v) for v in np.asarray(m, dtype=np.float64).reshape(-1)]
    d = m[0] * m[4] - m[1] * m[3]
    d = 1.0 / d if d != 0 else 0.0
    a11, a22 = m[4] * d, m[0] * d
    m[0] = a11
    m[1] *= -d
    m[3] *= -d
    m[4] = a22
    b1 = -m[0] * m[2] - m[1] * m[5]
    b2 = -m[3] * m[2] - m[4] * m[5]
    m[2], m[5] = b1, b2
    return m


def _table(device):
    key = str(device)
    if key not in _ITAB:
        _ITAB[key] = torch.from_numpy(lanczos4_table()).to(device)
    return _ITAB[key]


def _maps(inv, h, w, device):
    """Integer tap origin (sx, sy) and table row per destination pixel, as cv::WarpAffineInvoker."""
    f64 = torch.float64
    xs = torch.arange(w, device=device, dtype=f64)
    ys = torch.arange(h, device=device, dtype=f64)
    adelta = torch.round((inv[0] * xs) * AB_SCALE).long()
    bdelta = torch.round((inv[3] * xs) * AB_SCALE).long()
    x0 = torch.round((inv[1] * ys + inv[2]) * AB_SCALE).long() + ROUND_DELTA
    y0 = torch.round((inv[4] * ys + inv[5]) * AB_SCALE).long() + ROUND_DELTA
    shift = AB_BITS - INTER_BITS
    big_x = (x0[:, None] + adelta[None, :]) >> shift
    big_y = (y0[:, None] + bdelta[None, :]) >> shift
    sx = (big_x >> INTER_BITS) - 3
    sy = (big_y >> INTER_BITS) - 3
    row = (big_y & (TAB - 1)) * TAB + (big_x & (TAB - 1))
    return sx.reshape(-1), sy.reshape(-1), row.reshape(-1)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _lanczos4_kernel(src, dst, sx_ptr, sy_ptr, row_ptr, table, npix, h, w, BLOCK: tl.constexpr):
        frame = tl.program_id(1)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = offs < npix
        sx = tl.load(sx_ptr + offs, mask=live, other=0)
        sy = tl.load(sy_ptr + offs, mask=live, other=0)
        row = tl.load(row_ptr + offs, mask=live, other=0)
        base = src + frame.to(tl.int64) * npix
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for ky in tl.static_range(8):
            yy = sy + ky
            iny = live & (yy >= 0) & (yy < h)
            for kx in tl.static_range(8):
                xx = sx + kx
                inside = iny & (xx >= 0) & (xx < w)
                v = tl.load(base + yy * w + xx, mask=inside, other=0).to(tl.int32)
                wk = tl.load(table + row * 64 + (ky * 8 + kx), mask=live, other=0)
                acc += v * wk
        out = (acc + 16384) >> 15
        out = tl.minimum(tl.maximum(out, 0), 255)
        tl.store(dst + frame.to(tl.int64) * npix + offs, out.to(tl.uint8), mask=live)
except ImportError:  # pragma: no cover - torch fallback below stays exact
    triton = None


def warp_lanczos4(frames, forward_matrix, backend='auto'):
    """uint8 [F,H,W] (any device) warped by one forward 2x3 matrix -> uint8 [F,H,W] on the same device."""
    if frames.dtype != torch.uint8 or frames.ndim != 3:
        raise ValueError('warp_lanczos4 expects uint8 frames [F,H,W]')
    f, h, w = frames.shape
    device = frames.device
    inv = invert_affine(forward_matrix)
    sx, sy, row = _maps(inv, h, w, device)
    if backend in ('auto', 'triton') and triton is not None and device.type == 'cuda':
        src = frames.contiguous()
        dst = torch.empty_like(src)
        npix = h * w
        block = 256
        _lanczos4_kernel[(triton.cdiv(npix, block), f)](
            src, dst, sx.to(torch.int32).contiguous(), sy.to(torch.int32).contiguous(),
            row.to(torch.int32).contiguous(), _table(device), npix, h, w, BLOCK=block)
        return dst
    if backend == 'triton':
        raise RuntimeError('triton backend unavailable')
    weights = _table(device)[row]                      # [HW, 64] int32
    src = frames.reshape(f, h * w).to(torch.int32)
    acc = torch.zeros((f, h * w), dtype=torch.int32, device=device)
    for ky in range(8):
        yy = sy + ky
        inside_y = (yy >= 0) & (yy < h)
        base = yy.clamp(0, h - 1) * w
        for kx in range(8):
            xx = sx + kx
            inside = inside_y & (xx >= 0) & (xx < w)
            wk = torch.where(inside, weights[:, ky * 8 + kx], torch.zeros((), dtype=torch.int32, device=device))
            idx = base + xx.clamp(0, w - 1)
            acc += torch.index_select(src, 1, idx) * wk
    out = (acc + (1 << (COEF_BITS - 1))) >> COEF_BITS
    return out.clamp_(0, 255).to(torch.uint8).view(f, h, w)


def warp_bags(images, matrices, apply):
    """uint8 [B,K,3,H,W] bags; per-study forward matrices [B,2,3] (float64) and apply flags [B]."""
    if images.dtype != torch.uint8 or images.ndim != 5:
        raise ValueError('warp_bags expects uint8 bags [B,K,3,H,W]')
    out = torch.empty_like(images)
    b, k, c, h, w = images.shape
    for i in range(b):
        if bool(apply[i]):
            out[i] = warp_lanczos4(images[i].reshape(k * c, h, w), matrices[i].tolist()).view(k, c, h, w)
        else:
            out[i] = images[i]
    return out


def check_against_cv2(frames_np, forward_matrix, device='cuda'):
    """Return (#mismatching pixels, max |diff|) vs cv2 for uint8 [F,H,W] numpy frames."""
    import cv2
    h, w = frames_np.shape[-2:]
    ref = np.stack([cv2.warpAffine(f, np.asarray(forward_matrix, np.float64), (w, h), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT) for f in frames_np])
    got = warp_lanczos4(torch.from_numpy(np.ascontiguousarray(frames_np)).to(device), forward_matrix).cpu().numpy()
    diff = np.abs(ref.astype(np.int16) - got.astype(np.int16))
    return int((diff > 0).sum()), int(diff.max())


# ---------------------------------------------------------------------------------------------
# Loader / trainer wiring (``warp_device: gpu``). The dataset draws the rotation angle and scale
# with exactly the calls and order of native_aug.DegradeMixin._warp, builds the same forward matrix
# (cv2.getRotationMatrix2D) and ships it with the untouched uint8 bag; the trainer warps on the
# device right before the intensity LUT, i.e. at the position the CPU path warps. Evaluation bags
# never carry a matrix (train=False draws nothing), so evaluation is untouched.
# ---------------------------------------------------------------------------------------------
WARP_KEYS = {'warp_device'}
ORDER_KEYS = {'degrade_before_warp'}  # recipe key (not infrastructure): order of degradation vs the warp


class GpuWarpMixin:
    """Put in front of native_aug.with_degradation(...): defers the training Lanczos-4 warp to the GPU."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.gpu_warp = self.train and cfg.get('warp_device', 'cpu') == 'gpu' and bool(self.rotation or self.scale_jitter)
        if self.gpu_warp:
            if self.warp != 'lanczos4':
                raise ValueError('warp_device gpu implements warp_interpolation lanczos4 only')
            # The CPU path warps first and degrades the warped frames; with the warp deferred to the GPU the
            # (CPU, loader-side) degradation acts on the unwarped frames instead. The random draws keep their order
            # (angle, scale, then degradation p/scale), only the two image operations swap. That is a different
            # recipe, so it must be declared: ``degrade_before_warp: true`` (effnet512 runtime_b). Physically it
            # is the natural order: the resampling it imitates happened at corpus build time, before augmentation.
            if self.degrade_prob > 0 and not cfg.get('degrade_before_warp', False):
                raise ValueError('warp_device gpu would reorder warp and degradation (declare degrade_before_warp)')
            if not self.uint8_output:
                raise ValueError('warp_device gpu requires the uint8/LUT transfer path')
        self._pending = None

    def _warp(self, images):
        if not self.gpu_warp:
            return super()._warp(images)
        import random
        import cv2
        h = images.shape[-1]
        angle = random.uniform(-self.rotation, self.rotation)
        scale = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
        self._pending = cv2.getRotationMatrix2D(((h - 1) / 2, (h - 1) / 2), angle, scale)
        return images

    def _item(self, uid, row, selected):
        self._pending = None
        item = super()._item(uid, row, selected)
        if self.gpu_warp:
            matrix = self._pending
            item['warp_apply'] = torch.tensor(matrix is not None)
            item['warp_affine'] = torch.from_numpy(np.asarray(
                matrix if matrix is not None else [[1., 0., 0.], [0., 1., 0.]], dtype=np.float64).copy())
        self._pending = None
        return item


def with_gpu_warp(base):
    return type(base.__name__ + 'GpuWarp', (GpuWarpMixin, base), {})


def collate_with_warp(collate):
    def wrapped(items):
        batch = collate(items)
        flags = ['warp_affine' in item for item in items]
        if any(flags):
            if not all(flags):
                raise ValueError('warp matrices must be collated for all studies or none')
            batch['warp_affine'] = torch.stack([item['warp_affine'] for item in items])
            batch['warp_apply'] = torch.stack([item['warp_apply'] for item in items])
        return batch
    wrapped.__wrapped__ = collate
    return wrapped


def device_images_with_warp(shared):
    def device_images(batch, device, windows=slice(None)):
        if 'warp_affine' not in batch:
            return shared(batch, device, windows)
        from kneemri.coatnet_original_data import expand_intensity
        images = batch['images'][:, windows].to(device, non_blocking=True)
        images = warp_bags(images, batch['warp_affine'], batch['warp_apply'])
        return expand_intensity(images, batch['intensity_lut'].to(device, non_blocking=True))
    return device_images


# ---------------------------------------------------------------------------------------------
# runtime_c: training-only random Gaussian blur on the device (``blur_prob``, ``blur_sigma: [lo, hi]``).
# Per study: one Bernoulli(blur_prob) draw and, if blurred, sigma ~ U(lo, hi) (python ``random``, drawn in
# the loader after the warp/intensity draws); applied to the uint8 bag after the warp and before the LUT,
# i.e. where native_aug's CPU degradation sits. Separable kernel radius ceil(3 sigma), BORDER_REFLECT_101,
# round-to-nearest back to uint8. Evaluation bags carry no sigma (untouched).
# ---------------------------------------------------------------------------------------------
BLUR_KEYS = {'blur_prob', 'blur_sigma'}


def blur_settings(cfg):
    prob = float(cfg.get('blur_prob', 0.0))
    lo, hi = (float(v) for v in cfg.get('blur_sigma', [0.3, 1.0]))
    if not 0.0 <= prob <= 1.0 or not 0.0 < lo <= hi <= 3.0:
        raise ValueError('blur_prob must lie in [0,1] and blur_sigma in (0,3]')
    return prob, lo, hi


def gaussian_blur_u8(frames, sigma):
    """uint8 [F,H,W] -> uint8 [F,H,W], separable Gaussian, reflect-101 borders."""
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=frames.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    v = frames.float()[:, None]
    v = torch.nn.functional.pad(v, (radius, radius, 0, 0), mode='reflect')
    v = torch.nn.functional.conv2d(v, k.view(1, 1, 1, -1))
    v = torch.nn.functional.pad(v, (0, 0, radius, radius), mode='reflect')
    v = torch.nn.functional.conv2d(v, k.view(1, 1, -1, 1))
    return v[:, 0].round_().clamp_(0, 255).to(torch.uint8)


class GpuBlurMixin:
    """Put in front of GpuWarpMixin(...): draws the per-study blur after every other augmentation draw."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.blur_prob, self.blur_lo, self.blur_hi = blur_settings(cfg)
        self.gpu_blur = self.train and self.blur_prob > 0
        if self.gpu_blur and not self.uint8_output:
            raise ValueError('device blur requires the uint8/LUT transfer path')

    def _item(self, uid, row, selected):
        item = super()._item(uid, row, selected)
        if self.gpu_blur:
            import random
            sigma = random.uniform(self.blur_lo, self.blur_hi) if random.random() < self.blur_prob else 0.0
            item['blur_sigma'] = torch.tensor(float(sigma) if len(selected) else 0.0, dtype=torch.float64)
        return item


def with_gpu_augment(base):
    return type(base.__name__ + 'GpuAug', (GpuBlurMixin, GpuWarpMixin, base), {})


def collate_with_augment(collate):
    inner = collate_with_warp(collate)

    def wrapped(items):
        batch = inner(items)
        flags = ['blur_sigma' in item for item in items]
        if any(flags):
            if not all(flags):
                raise ValueError('blur sigmas must be collated for all studies or none')
            batch['blur_sigma'] = torch.stack([item['blur_sigma'] for item in items])
        return batch
    wrapped.__wrapped__ = collate
    return wrapped


def device_images_with_augment(shared):
    def device_images(batch, device, windows=slice(None)):
        if 'warp_affine' not in batch and 'blur_sigma' not in batch:
            return shared(batch, device, windows)
        from kneemri.coatnet_original_data import expand_intensity
        images = batch['images'][:, windows].to(device, non_blocking=True)
        if 'warp_affine' in batch:
            images = warp_bags(images, batch['warp_affine'], batch['warp_apply'])
        if 'blur_sigma' in batch:
            b, k, c, h, w = images.shape
            for i, sigma in enumerate(batch['blur_sigma'].tolist()):
                if sigma > 0:
                    images[i] = gaussian_blur_u8(images[i].reshape(k * c, h, w), sigma).view(k, c, h, w)
        return expand_intensity(images, batch['intensity_lut'].to(device, non_blocking=True))
    return device_images
