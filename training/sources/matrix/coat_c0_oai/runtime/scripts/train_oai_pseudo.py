#!/usr/bin/env python3
"""CoAtNet C0 original route (train_soft.py, 10-17 G07/G08) + OAI kd50 mixing + per-update weight EMA.

final_matrix_20261020, Gu handoff. Launch with torch.distributed.run (window-parallel encoder, every rank
gets the identical study batch, effective batch 4 x 2 = 8 studies), like the 10-17 G07/G08 recipes.
Install order: soft objective (mass-preserved BCE) -> portable full-refit adapter (OAI-aware copy of the
10-17 tools/portable_refit.py) -> kneemri.oai_pseudo (after the objective, as in every OAI family) ->
kneemri.ema_hook. Without oai_* keys and with ema_decays [] this is exactly the 10-17 T037/T043 path.
The C0 trainer predates two arguments that oai_pseudo passes through: the bag class's ``uint8_output`` (C0Bags
accepts False only, this route never used uint8 transfer) and make_loader's ``pin_memory`` (dropped, the C0
loader keeps its own pinning). Nothing else changes.
"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[2]  # family root: train_soft.py, soft_objective.py, portable_refit_oai.py
sys.path[:0] = [str(HERE / 'runtime/src'), str(HERE / 'runtime/scripts'), str(HERE)]
import train_coatnet_ablation as trainer  # noqa: E402
from soft_objective import install as install_soft  # noqa: E402
import portable_refit_oai  # noqa: E402
from kneemri import ema_hook, oai_pseudo  # noqa: E402

import kneemri.coatnet_original_data as original_data  # noqa: E402

_Bags = original_data.CoAtNetOriginalNeighborBags


class C0Bags(_Bags):
    def __init__(self, cfg, all_ids, ids, labels, train, uint8_output=False):
        if uint8_output:
            raise ValueError('the C0 data module has no uint8 transfer')
        super().__init__(cfg, all_ids, ids, labels, train)
        self.uint8_output = False


_make_loader = trainer.make_loader


def make_loader(dataset, cfg, device, train=False, generator=None, pin_memory=True):
    return _make_loader(dataset, cfg, device, train, generator)


if __name__ == '__main__':
    cfg = yaml.safe_load(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
    original_data.CoAtNetOriginalNeighborBags = C0Bags
    trainer.make_loader = make_loader
    install_soft(trainer)
    portable_refit_oai.install(trainer, cfg, dict(id=cfg['validation_role']))
    oai_pseudo.install(trainer)
    ema_hook.install(trainer)
    trainer.main()
