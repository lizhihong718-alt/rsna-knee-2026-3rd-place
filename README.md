# RSNA Knee 2026 — DataTech Club, 3rd place

**Release in progress (2026-11-01):** source code is public. The weights upload
and fresh Kaggle notebook validation are still running/pending; the asset links
below are the intended release locations until verification is recorded.

This is the release of **G36-Gold**, the submission selected for our third-place
result in RSNA Knee Abnormality Detection 2026. The final selected submissions
were scored separately: G36-Gold scored public **0.968**, private **0.959**;
A21GT scored public 0.968, private 0.958. A21GT was submitted later and is not
the winning model released here. Scores use Kaggle's displayed precision.

The winning submission is **57527835**, script version **361161492**:
[original notebook](https://www.kaggle.com/code/yeeelzh/g36-gold-final-20261022-score?scriptVersionId=361161492).
Its original frozen manifest SHA256 is
`07e007aeab7ce5bfdad53b52d5180f3ceba708256d0fcc0d24332f08ab487916`.
The original notebook may remain private because its package and outputs contain
research metadata. The clean reproduction release preserves its inference code,
model tensors, input processing and blending.

## Release assets

- [Model weights](https://www.kaggle.com/datasets/yeeelzh/rsna-knee-2026-third-place-g36-weights): all 36 final inference checkpoints, runtime archive and SHA256 manifest.
- [Reproduction notebook](https://www.kaggle.com/code/yeeelzh/rsna-knee-2026-third-place-g36-reproduction).
- [Team solution](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/3rd-place-solution-rsna-knee-abnormality-detectio).
- `model_manifest.json`: exact model IDs, seeds, original and released weight hashes,
  readouts and Gold-refit status. `inference/members.json` contains every
  architecture, input variant and per-class mixing coefficient.
- `training/recipes.json` and `training/configs/`: the 36 matched training recipes
  and their separate, preserved source versions.

## Run inference

Use Linux, Python 3.11, PyTorch 2.6.0 + CUDA 12.4 and torchvision 0.21.0.
Install the CUDA wheels, followed by the pinned requirements:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python verify.py --weights /path/to/downloaded/weights
CUDA_VISIBLE_DEVICES=0,1 python infer.py \
  --data /path/to/rsna-knee-abnormality-detection \
  --weights /path/to/downloaded/weights \
  --out /path/to/private-output/submission.csv
```

The data directory must contain `test.csv`, `test_series.csv`, and `test_series/`
in the competition layout. Obtain competition data from Kaggle under its terms.
No reports or OAI data are needed for inference. Keep generated predictions and
logs private. This code generates predictions and does not submit them to Kaggle.

The preserved lane plan uses **two NVIDIA T4 GPUs with 16 GB each**, eight
process lanes sharing those GPUs, three CPU preprocessing workers and four DICOM
reader threads. It decodes each study once and retains checkpoint-specific native
224/384/512 or legacy 384 input processing. FP16 and per-lane chunk sizes are
preserved. Use at least 32 GB host RAM and 25 GB free working storage for weights,
unpacking and bounded temporary caches; 64 GB RAM is a practical recommendation.

The original Kaggle visible-test execution completed in **5m29s on T4 x2** for
three visible studies, including startup. Its recorded scoring compute time was
202.98 seconds. The original 1,300-study budget forecast was 7.38 hours including
margin; this is an estimate, not a measured hidden-test runtime. See
`VALIDATION.md` for fresh release verification. `Dockerfile` supplies an
alternative build environment; a successful image build is not asserted without
an accompanying validation record.

## Reproduce training

Training is separate from inference and is never started by installation or
verification. The recipes use fixed 25-epoch endpoints, 32 EMA(0.999) readouts and
four Raw readouts. **27 models include Gold58 in training; nine retain their
original non-Gold weights.** Gold58 evaluations of refits are in-sample and are
not independent validation. No new training was performed for this release.

```bash
python train.py --id Z002 --paths paths.local.json \
  --out /path/to/private-runs/Z002 --prepare-only
CUDA_VISIBLE_DEVICES=0 python train.py --id Z002 --paths paths.local.json \
  --out /path/to/private-runs/Z002-new
```

Supply a JSON object mapping each referenced key in `training/input_assets.json`
to the corresponding authorized local file or cache. Each recipe identifies its
inputs explicitly as `assets/<key>`; file/cache-manifest hashes are checked before
training. The same basenames can refer to different label versions, so do not
substitute by filename alone. `--prepare-only` resolves inputs and writes the
requested configuration without starting a trainer.

The frozen training inputs include RSNA N/R soft-label snapshots, protected
report-group folds, orientation metadata, original/native MRI caches, public
backbone initializers, OAI MRI caches, mapped OAI targets/weights and, for coupled
models, native ordinal targets and the frozen auxiliary loss scale. Training uses
H20-class GPUs; individual 25-epoch recipes take several to tens of GPU-hours
depending on backbone, resolution and OAI exposure. The large legacy CoAtNet
training route supports window-parallel `torch.distributed.run`; its original
recipe uses **four GPUs / four distributed processes**, batch 4 and accumulation
2 (effective study batch 8; the ranks share the same study batch and divide its
windows). `train.py` launches that original distributed entry automatically for
`FM_coat_o384_N_kall`; the other 35 recipes use one process each.
Inference resource requirements do not imply that T4 is sufficient for every
original training recipe.

**Data availability limitation:** this release does not redistribute MRI,
reports, subject IDs, per-subject labels, folds or OAI/NDA files. Training code and
configurations are supplied, but an exact rerun requires the hashed frozen inputs
through an authorized route. A new LLM annotation run is not guaranteed to
reconstruct the same targets. The released weights allow inference reproduction
without those training inputs. This distinction is deliberate and must not be
described as a fully public, bitwise training reproduction.

## Fixed ensemble

For each class, `B = 0.5 * mean(Mature12) + 0.5 * mean(Matrix20)` using sigmoid
probabilities. Nine classes use B directly. Lateral OA, PF OA and Synovitis use
`0.5 * B + 0.5 * mean(Expert8)`. Four models occur in both Mature12 and Expert8;
they are executed once and retain both contributions. There are 36 unique
checkpoints. No percentile-rank conversion or new blend fitting is applied.

## Attribution and licenses

Original solution authors: Zhengru Li, Zhihong Li, Bufan Li, Guoqin Gu and Mutian
Hong, DataTech Club. Our contributions use **CC-BY-NC-4.0**, the competition's
specified winner license. Third-party components retain their licenses.

- DINOv3: Meta, DINOv3 License, supplied in `licenses/DINOv3_LICENSE.md`.
- PyTorch Image Models (timm): Ross Wightman and contributors, Apache-2.0,
  supplied in `licenses/timm_LICENSE`; model-specific upstream rights remain.
- BiT, ConvNeXt and CoAtNet pretrained initializers are identified by architecture
  and hashes in the training input manifests; they are obtained separately.
- The Raptor-compatible input/model reference credits Dread Development /
  Johnathan Wagner's [Kaggle notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model).
  Version 7 is released under Apache-2.0; its source attribution is retained.
  No public task-trained Raptor checkpoint
  is included in the released 36-model ensemble.
- RSNA competition data retain the [MIRA terms](https://www.rsna.org/mira-license).
- Selected training recipes used the Osteoarthritis Initiative (OAI), obtained
  through authorized access via the NIMH Data Archive (NDA). OAI/NDA data are
  not redistributed. See [OAI](https://nda.nih.gov/oai/) for authorized access.
  The investigators who collected OAI data do not endorse this solution.

Use is for research under the applicable licenses. This release is not a clinical
device or a claim of diagnostic safety.

The Dataset's SHA-pinned `source.zip` is the frozen offline inference archive.
Use this repository's current `train.py`, recipes and instructions for training,
including the explicit four-process launcher for the legacy CoAtNet recipe.
