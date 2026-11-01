#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import train_cnx_dino384 as base
from kneemri import oai_pseudo
def setup(path):
    t=base.trainer
    base.install_depth(path)
    base.install_compile()
    t.source_receipt=base.source_receipt
    base.install_gold(t)
    base.install_fullstrength(t)
    old=t.normalize_config
    t.normalize_config=lambda config:base.check_knobs(old(config))
    oai_pseudo.install(t)
    return t
if __name__=='__main__':setup(sys.argv[sys.argv.index('--config')+1]).main()
