#!/usr/bin/env python3
"""train_resnet_n384.py (E1 = R2 + EMA trainer, runtime_up model_input_size hook) plus kneemri.oai_pseudo.

Without any oai_pseudo key (oai_index / init_checkpoint ...) this is exactly the
train_resnet_n384 entry point.
"""
import train_resnet_n384 as n384
import kneemri.coatnet_original_data as original_data
from kneemri import native_aug, oai_pseudo
from kneemri import resnet_depth_e2e as e2e
from resnet_fullstrength import install


def main():
    trainer = n384.train_resnet.trainer
    trainer.MODEL_KEYS.update(e2e.E2E_MODEL_KEYS | {'model_input_size'})  # final_matrix: native224 up320 hook
    trainer.DATA_KEYS.update(e2e.E2E_DATA_KEYS | native_aug.DEGRADE_KEYS)
    trainer.build_model = n384.build_model
    trainer.source_receipt = n384.source_receipt
    install(trainer)
    original_data.CoAtNetOriginalNeighborBags = native_aug.with_degradation(e2e.DepthSeriesBags)
    oai_pseudo.install(trainer)
    trainer.main()


if __name__ == '__main__':
    main()
