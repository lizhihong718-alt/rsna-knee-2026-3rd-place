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

Fresh public-package execution and publication checks are recorded below only
after they actually complete. No new competition submission or retraining is
part of release validation.
