"""Two-view 224 bags: every window is seen at 140 mm (native224) and at a centred zoom.

224 px over the 140 mm crop is 0.625 mm/px; the classes it loses against 384 are the
fine-detail ones (Lateral Meniscus, Lateral OA, PF OA). The zoom view crops the
centred ``fov_mm`` of the same window from the single-resample native384 corpus and
AREA-downsamples it to 224 (only ever downsampled, never upscaled: 100 mm -> 274 px
-> 224 px, 0.446 mm/px). Both views share the window's slices, canonical flip,
geometric draw, intensity table and metadata; the bag holds view v of window i at
v*K + i. Evaluation and inference keep every window of both views.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from kneemri.coatnet_original_data import CoAtNetOriginalNeighborBags

ZOOM_KEYS = {'zoom_view'}


class ZoomViewBags(CoAtNetOriginalNeighborBags):
    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
        super().__init__(cfg, all_ids, ids, labels, train, uint8_output)
        spec = cfg['zoom_view']
        self.zoom_root = Path(spec['corpus'])
        manifest = json.loads((self.zoom_root / 'manifest.json').read_text())
        pre = manifest['preprocessing']
        if manifest['study_uids'] != list(all_ids) or float(pre['crop_fov_mm']) != 140.0:
            raise ValueError('zoom corpus must be the same 140 mm crop of the same studies')
        if pre.get('spatial_resample_count') != 1:
            raise ValueError('zoom corpus must be a single-resample native corpus')
        shape = manifest['shape']
        if shape[:3] != list(self.manifest['shape'][:3]):
            raise ValueError('zoom corpus windows must align with the base corpus')
        source = int(shape[-1])
        self.zoom_px = int(round(source * float(spec['fov_mm']) / 140.0))
        if (source - self.zoom_px) % 2:
            self.zoom_px += 1  # symmetric margins keep the canonical mirror exact
        if self.zoom_px <= self.resolution:
            raise ValueError('zoom view must be downsampled to the training resolution')
        self.zoom_offset = (source - self.zoom_px) // 2
        self.views = 2
        self.zoom_windows = None

    def _open(self):
        super()._open()
        if self.zoom_windows is None:
            self.zoom_windows = np.load(self.zoom_root / 'windows.npy', mmap_mode='r')

    def __getstate__(self):
        state = super().__getstate__()
        state['zoom_windows'] = None
        return state

    def _zoom(self, raw):
        o, n, r = self.zoom_offset, self.zoom_px, self.resolution
        crop = raw[..., o:o + n, o:o + n].reshape(-1, n, n)
        out = np.stack([cv2.resize(np.ascontiguousarray(f), (r, r), interpolation=cv2.INTER_AREA)
                        for f in crop])
        return out.reshape(raw.shape[0], 3, r, r)

    def _read(self, row, selected):
        base, _ = super()._read(row, selected)
        zoom = self._canonical(row, selected, self._zoom(np.asarray(self.zoom_windows[row, selected])))
        return np.concatenate([base, zoom]), np.concatenate([selected, selected])
