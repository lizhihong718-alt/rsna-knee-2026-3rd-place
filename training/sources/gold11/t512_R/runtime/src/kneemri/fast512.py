"""Training-only torch.compile of the backbone feature extractor (runtime_b, throughput only).

Measured at 512 px, bf16, 48 images/step on an H20 shared with a running arm:
ConvNeXt-S 0.595 -> 0.428 s/step, DINOv3 ViT-B/16 1.013 -> 0.733 s/step (x1.38 both).

* Only the training forward is compiled: when grad is disabled or the module is in eval mode the
  original eager ``forward_features`` runs, so Gold58 monitoring, saved checkpoints (state_dict keys
  are untouched; the wrapper is an instance attribute, not a submodule) and DICOM inference are the
  exact eager path.
* Whole-series bags (E2E_D) give a different window count every step. Rows are padded to a multiple
  of ``bucket`` with copies of the last image and the padded outputs dropped, so the compiled graph
  sees a few static shapes. ConvNeXt (LayerNorm) and ViT (LayerNorm) have no batch statistics:
  every output row depends on its own input only, and the dropped rows receive no gradient.
"""
import torch

from kneemri.coatnet_model import DinoPatchBackbone


def install(model, bucket=16):
    backbone = model.backbone
    target = backbone.base if isinstance(backbone, DinoPatchBackbone) else backbone
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in target.modules()):
        raise ValueError('row padding is only exact for backbones without batch statistics')
    if bucket < 1:
        raise ValueError('compile_bucket must be >= 1')
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    eager = target.forward_features
    compiled = torch.compile(eager, dynamic=False)

    def forward_features(x):
        if not (torch.is_grad_enabled() and target.training):
            return eager(x)
        rows = x.shape[0]
        pad = (-rows) % bucket
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], 0)
        x = x.contiguous(memory_format=torch.channels_last)
        out = compiled(x)
        return out[:rows] if pad else out

    target.forward_features = forward_features
    return model
