#!/usr/bin/env python3
import train_resnet_n384 as n384
import kneemri.coatnet_original_data as original_data
from kneemri import native_aug,oai_pseudo,squeeze
from kneemri import resnet_depth_e2e as e2e
from resnet_fullstrength import install

def setup():
    trainer=n384.train_resnet.trainer
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS)
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS|native_aug.DEGRADE_KEYS)
    trainer.build_model=n384.build_model
    trainer.source_receipt=n384.source_receipt
    install(trainer)
    original_data.CoAtNetOriginalNeighborBags=native_aug.with_degradation(e2e.DepthSeriesBags)
    oai_pseudo.install(trainer)
    squeeze.install(trainer)
    from kneemri import native_aux
    native_aux.install(trainer)
    return trainer

if __name__=='__main__': setup().main()
