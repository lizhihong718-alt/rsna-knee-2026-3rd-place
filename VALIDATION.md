# Release validation

Verified from the final selected Kaggle submissions on 2026-11-01:
G36-Gold (57527835, script 361161492) and A21GT were selected; G36-Gold's 0.959
private score is higher than A21GT's 0.958. The final selection is not inferred
from submission recency or the rounded maximum of all historical scores.

Completed before publication:

- All 36 original weight files match the winning bundle's SHA256 pins.
- 33 existing safetensors files retain their original bytes. Three `.pt` model
  dictionaries were exported to safetensors; every tensor key, dtype, shape and
  value matches exactly. Training metadata were excluded from those exports.
- All 36 released weights strictly load into their original inference model
  constructors on CPU. Architecture and resolution match for every member.
- Preserved inference source files match their original hashes. Only inert
  private configuration paths, weight filenames/serialization and package
  bookkeeping were changed. DICOM geometry, normalization, sampling, FP16 policy,
  model equations and fusion code were retained.
- Runtime archive verification and all 12 fixed class-weight formulas pass.
  Membership includes all 36 models and each class's coefficients sum to one.
- All 36 original training entries import and reach their argument parser's help
  successfully in the pinned CPU environment. No optimizer or training step ran.
- All 36 training configurations resolve against the existing authorized local
  input paths and pass their file/cache-manifest hash checks in prepare-only mode.
- Source/configuration scans find no cluster/user private paths or credential
  literals. Weight headers contain only model identifiers and provenance hashes.
  DICOM transfer-syntax UIDs in decoder code are standards identifiers, not
  patient identifiers.

Existing winning-package evidence, not a new release run:

- Local inference completed for all 36 models on 58 studies and on the three
  visible test studies without missing members or degraded fallback.
- The original Kaggle notebook completed on T4 x2. Local/Kaggle maximum
  probability differences were 0.00067703 for a member and 0.00002876 for the
  ensemble, within the unchanged 0.002 gate.
- Gold58 was used only to check execution and parity for the Gold refits;
  these results establish artifact correctness, not held-out generalization.

## Fresh release execution on 2026-11-01

The repackaged public weights and offline source completed development execution
on **T4 x2**, scoring all **36 models × 3 visible test studies**. Every weight
SHA, member ID, output column, study order and probability range passed. Coverage
was 100%, with no dropped members, preprocessing failures or degraded fallback.

Compared with the original winning script version 361161492 on the same visible
studies, maximum absolute probability differences were **0.0004236497** for a
member and **0.0000377388** for the ensemble. Both pass the original unchanged
**0.002** gate. These are inference-correctness checks, not new leaderboard or
clinical-validation results.

The fresh run used Python **3.13.15**, torch **2.11.0+cu128**, torchvision
**0.26.0+cu128**, timm **1.0.29**, numpy **2.1.3**, pandas **2.3.3**, pydicom
**3.0.2**, and PyYAML **6.0.3**. This observed Kaggle environment is distinct from
the Python 3.11 / torch 2.6 reference training/container environment. Both the
runtime environment and per-worker peak GPU allocations are retained in
`validation/development_reproduction_audit.json`.

Elapsed execution/verification was **293.39 seconds** after source materialization;
the recorded inference portion was **221.55 seconds**, including **41.80 seconds**
of loading and **179.75 seconds** of scoring. Kaggle startup/artifact overhead is
separate. No retraining or new competition submission was performed.

The public notebook's final execution and publication readbacks will be recorded
after completion. Its output policy publishes only aggregate verification;
per-study predictions and logs remain private.
