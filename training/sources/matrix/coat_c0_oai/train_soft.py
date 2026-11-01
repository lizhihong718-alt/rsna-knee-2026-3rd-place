import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime/scripts'))
sys.path.insert(0,str(HERE/'runtime/src'))
import train_coatnet_ablation as trainer
from soft_objective import install, tests
if __name__=='__main__':
    if '--objective-tests' in sys.argv:tests(trainer)
    else:
        install(trainer)
        trainer.main()
