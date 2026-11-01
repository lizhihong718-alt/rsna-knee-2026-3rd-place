#!/usr/bin/env python3
"""Train one fresh ResNet with Astra v9 full-strength supervision."""
import train_resnet
from resnet_fullstrength import install


def main():
    # Preserve the ResNet model/provenance hooks before installing the objective.
    trainer = train_resnet.trainer
    trainer.build_model = train_resnet.build_model
    trainer.source_receipt = train_resnet.source_receipt
    from kneemri.gold_cohort import install as install_gold
    install_gold(trainer)  # before the objective, whose constants check must see Gold
    install(trainer)
    trainer.main()


if __name__ == '__main__':
    main()
