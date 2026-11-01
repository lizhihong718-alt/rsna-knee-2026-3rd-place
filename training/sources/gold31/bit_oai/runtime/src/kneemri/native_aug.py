"""Training-only resolution degradation for single-resample (native) corpora.

native384_crop140 differs from the legacy double-resampled 384 cache almost only in
mid/high spatial frequencies: 0.25-0.5 Nyquist power x1.24, 0.5-0.8 Nyquist x1.6,
Immerkaer noise x1.35, rising to x1.5 on studies whose native 140 mm crop exceeds
550 px (analysis/image_stats.py). The native-512 ResNet regressed most on exactly
those fine-resolution studies, i.e. the extra band carries scanner-specific noise as
well as detail. Here a training bag is, with probability ``degrade_prob``, low-passed
before the intensity transform, so the encoder must not rely on that band while
evaluation and inference keep the native pixels unchanged.

Modes (one draw per study bag, like the rotation/scale draw):
  resample: INTER_AREA down to a random fraction of the side, INTER_LINEAR back
            (the same two operations that produced the legacy cache);
  blur:     isotropic Gaussian with a random sigma in pixels.
The transform is uint8 -> uint8, so the compact uint8/LUT transfer path stays exact.

``warp_interpolation``: the legacy rotation/scale warp (always on in training, never
in evaluation) uses INTER_LINEAR, which leaves native384 only ~0.65x of its 0.5-1.0
Nyquist power (cnx_dino_384 analysis/spectrum_384.json); evaluation windows keep 1.0x.
'lanczos4' (~1.0x) or 'cubic' (~0.9x) removes that train/eval sharpness gap. 'linear'
(default) calls the legacy method itself, so existing recipes are bit-identical.
"""
import random

import numpy as np

DEGRADE_KEYS = {'degrade_prob', 'degrade_mode', 'degrade_scale', 'degrade_sigma', 'warp_interpolation',
                'eval_degrade_scale'}
MODES = {'resample', 'blur'}
WARPS = {'linear', 'cubic', 'lanczos4'}


def _check(cfg):
    p = float(cfg.get('degrade_prob', 0.0))
    if not 0.0 <= p <= 1.0:
        raise ValueError('degrade_prob must lie in [0, 1]')
    mode = cfg.get('degrade_mode', 'resample')
    if mode not in MODES:
        raise ValueError(f'degrade_mode must be one of {sorted(MODES)}')
    lo, hi = cfg.get('degrade_scale', [0.6, 0.95])
    if not 0.25 <= lo <= hi <= 1.0:
        raise ValueError('degrade_scale must satisfy 0.25 <= lo <= hi <= 1')
    slo, shi = cfg.get('degrade_sigma', [0.4, 1.2])
    if not 0.0 < slo <= shi <= 4.0:
        raise ValueError('degrade_sigma must satisfy 0 < lo <= hi <= 4')
    if cfg.get('warp_interpolation', 'linear') not in WARPS:
        raise ValueError(f'warp_interpolation must be one of {sorted(WARPS)}')
    return p, mode, (float(lo), float(hi)), (float(slo), float(shi))


def degrade(images, mode, value):
    """uint8 [K,3,H,W] -> uint8 of the same shape."""
    import cv2
    cv2.setNumThreads(1)
    h, w = images.shape[-2:]
    flat = np.ascontiguousarray(images).reshape(-1, h, w)
    if mode == 'resample':
        sh, sw = max(8, int(round(h * value))), max(8, int(round(w * value)))
        out = [cv2.resize(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), (w, h),
                          interpolation=cv2.INTER_LINEAR) for f in flat]
    else:
        out = [cv2.GaussianBlur(f, (0, 0), sigmaX=value, sigmaY=value) for f in flat]
    return np.stack(out).reshape(images.shape)


class DegradeMixin:
    """Mix in front of an original-neighbour bag class; evaluation is untouched."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.degrade_prob, self.degrade_mode, self.degrade_scale, self.degrade_sigma = _check(cfg)
        self.warp = cfg.get('warp_interpolation', 'linear')
        # Diagnostic only (never in a training recipe): deterministic resample of eval windows.
        self.eval_degrade = cfg.get('eval_degrade_scale')

    def _warp(self, images):
        """The legacy rotation-then-scale draw and warp, with a selectable interpolation."""
        if self.warp == 'linear' or not (self.train and (self.rotation or self.scale_jitter)):
            return super()._geometric(images)
        import cv2
        cv2.setNumThreads(1)
        flag = {'cubic': cv2.INTER_CUBIC, 'lanczos4': cv2.INTER_LANCZOS4}[self.warp]
        h = images.shape[-1]
        angle = random.uniform(-self.rotation, self.rotation)
        scale = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
        affine = cv2.getRotationMatrix2D(((h - 1) / 2, (h - 1) / 2), angle, scale)
        flat = images.reshape(-1, h, h)
        images = np.stack([cv2.warpAffine(frame, affine, (h, h), flags=flag, borderMode=cv2.BORDER_CONSTANT)
                           for frame in flat])
        return images.reshape(-1, 3, h, h)

    def _geometric(self, images):
        images = self._warp(images)
        if not self.train and self.eval_degrade:
            return degrade(images, 'resample', float(self.eval_degrade))
        if not self.train or self.degrade_prob <= 0 or random.random() >= self.degrade_prob:
            return images
        lo, hi = self.degrade_scale if self.degrade_mode == 'resample' else self.degrade_sigma
        return degrade(images, self.degrade_mode, random.uniform(lo, hi))


def with_degradation(base):
    return type(base.__name__ + 'Degraded', (DegradeMixin, base), {})
