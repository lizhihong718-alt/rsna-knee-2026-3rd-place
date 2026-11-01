#!/usr/bin/env python3
"""train_coatnet_n384.py + kneemri.oai_pseudo (final matrix 20261020, Gu handoff).

Installs the OAI mixing hooks at the start of trainer.main(), after every hook that train_coatnet_n384.py itself
installs (full strength, Gold monitor, bag classes), the same order as oai_weak3's train_oai_pseudo.py.
Without any oai_* config key the hooks are inert and this is exactly train_coatnet_n384.py.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import train_coatnet_ablation as trainer  # noqa: E402
from kneemri import oai_pseudo  # noqa: E402

_main = trainer.main


def main():
    oai_pseudo.install(trainer)
    _main()


trainer.main = main
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'scripts' / 'train_coatnet_n384.py'), run_name='__main__')
