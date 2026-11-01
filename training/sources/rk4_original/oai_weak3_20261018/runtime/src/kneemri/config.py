"""Typed access to the preprocessing config (configs/preprocess.yaml).

Modules take plain dataclasses/dicts — never the raw YAML — so every stage is
unit-testable without a config file on disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SlotKey:
    plane: str
    fluid_sensitive: int
    fat_suppression: int | None = None  # None = any (train has fs==fluid; test may not)


@dataclass
class SelectionComponent:
    name: str
    weight: float
    params: dict[str, Any] = field(default_factory=dict)


def _default_slot_definitions() -> dict[str, list[SlotKey]]:
    """The canonical 6-slot knee MRI layout."""
    return {
        "sagittal_fluid_fs": [SlotKey("Sagittal", 1)],
        "coronal_fluid_fs": [SlotKey("Coronal", 1)],
        "axial_fluid_fs": [SlotKey("Axial", 1)],
        "sagittal_nonfluid": [SlotKey("Sagittal", 0)],
        "coronal_nonfluid": [SlotKey("Coronal", 0)],
        "axial_nonfluid": [SlotKey("Axial", 0)],
    }


def _default_selection() -> list[SelectionComponent]:
    """Canonical selection rule — MUST mirror configs/preprocess.yaml so a
    default-constructed config behaves exactly like the shipped one."""
    return [
        SelectionComponent("metadata_completeness", 2.0,
                           {"geometry": 0.5, "thickness": 0.2,
                            "spacing_between": 0.2, "sequence_meta": 0.1}),
        SelectionComponent("orientation_purity", 2.0),
        SelectionComponent("slice_count", 1.0, {"reference": 30}),
        SelectionComponent("coverage_mm", 1.0, {"reference": 150}),
        SelectionComponent("in_plane_resolution", 1.0, {"reference_mm": 0.4}),
        SelectionComponent("sequence_suitability", 1.0),
    ]


@dataclass
class SlotsConfig:
    definitions: dict[str, list[SlotKey]] = field(default_factory=_default_slot_definitions)
    selection: list[SelectionComponent] = field(default_factory=_default_selection)
    # how many ranked candidate series to KEEP per slot. 1 = Top-1 baseline;
    # K>1 lets the Dataset/model consume multiple series per slot (extra
    # candidates ranked by the same deterministic scoring rule and stored as
    # rank s1..s{K-1} alongside the Top-1).
    k_series_per_slot: int = 1


@dataclass
class GeometryConfig:
    # order matters: with no geometry at all, InstanceNumber rank disagreed
    # with physical order less often than SliceLocation in this corpus
    # (38.7% vs 58.9%) — both preserve adjacency, neither knows direction
    fallbacks: list[str] = field(default_factory=lambda: ["instance_number", "slice_location"])
    max_normal_deviation_deg: float = 5.0
    duplicate_z_tol_mm: float = 0.05
    min_slices: int = 2


@dataclass
class LateralityConfig:
    normalize: bool = True
    target: str = "lateral"
    min_axis_alignment: float = 0.5


@dataclass
class PreprocessConfig:
    crop_mode: str = "physical"  # full: preserve acquired FOV with physical square padding
    crop_fov_mm: float = 150.0
    output_size: int = 384
    intensity_percentile_low: float = 1.0
    intensity_percentile_high: float = 99.0
    intensity_over: str = "foreground"
    resize_interpolation: str = "auto"


@dataclass
class SamplingConfig:
    mode: str = "uniform_index"
    slices_per_slot: int = 32
    extent_mm: float | None = None
    neighbor_offsets: tuple[int, ...] = (-1, 0, 1)
    pad_mode: str = "edge"


@dataclass
class OutputConfig:
    dtype: str = "uint8"
    compression: bool = False
    include_metadata_json: bool = True


@dataclass
class DatasetConfig:
    """Training-side view of the preprocessed npz (src/kneemri/dataset.py)."""
    processed_dir: str = "data/processed/train"
    labels_csv: str = "data/ref_labels/llm_labels_v4_blend.csv"
    # expert-labeled table (58 gold studies; 12 label cols non-null there)
    train_csv: str = "data/train.csv"
    # model input resolution; the npz master copy stays at output_size (384)
    resize_to: int = 224
    # study-level cap; series beyond this per slot are dropped (already capped
    # at preprocessing time by slots.k_series_per_slot)
    max_series_per_slot: int | None = None
    # DINO feature cache (section 11): aggregator training never runs DINO
    cache_dir: str = "data/cache/dino_vits"
    max_series_per_study: int | None = 6
    centers_per_series: int | None = None
    augment: bool = False
    rotation_deg: float = 5.0
    scale_jitter: float = 0.05
    intensity_jitter: float = 0.1
    max_neighbor_gap_ratio: float = 1.8
    require_schema: int = 0
    require_targets: bool = False
    channel_mode: str = "adjacent"  # adjacent 2.5D | repeat center (2D control)
    train_centers_per_series: int | None = None
    position_frame: str = "physical"  # legacy patient-normal direction | canonical stored-stack progress


@dataclass
class EncoderConfig:
    # timm (preferred, sections 10) or tiny_conv (tests/CPU sanity)
    name: str = "timm"
    # exact timm model name; resolved against timm.list_models('*dinov3*')
    # with vit_small_patch16_dinov3 as the default preference — never a
    # silent fallback to a different architecture
    backbone_name: str = "vit_small_patch16_dinov3"
    pretrained: bool = True
    freeze_backbone: bool = True
    # progressive unfreezing: 0 | 2 | 4 | ... ViT blocks (from the top) or 'all'
    unfreeze_last_n_blocks: int | str = 0
    image_size: int = 224
    weights: str | None = None
    feature_pool: str = "cls"  # cls | cls_patch_mean (image training only)
    mean: list[float] | None = None
    std: list[float] | None = None


@dataclass
class SeriesTransformerConfig:
    layers: int = 2
    heads: int = 8
    dropout: float = 0.1


@dataclass
class StudyTokensConfig:
    """Section 14: what the pathology decoder sees after the within-series
    transformer. b (default) keeps every slice token [B, S*K*N, D] so queries
    can point at a specific series AND slice region; a pools each series into
    one token [B, S*K, D] (a series-level average)."""
    mode: str = "b"


@dataclass
class PathologyDecoderConfig:
    """Section 15: 12 learned pathology queries cross-attending MRI tokens."""
    num_queries: int = 12
    layers: int = 2
    heads: int = 8
    dropout: float = 0.1
    # optional self-attention among the 12 queries (default off)
    query_self_attention: bool = False


@dataclass
class LossConfig:
    """Section 18/19: weighted soft-label BCE + optional pairwise ranking."""
    gold_weight: float = 1.0
    pseudo_default_weight: float = 0.7
    lambda_rank: float = 0.1
    normalization: str = "weight_sum"  # legacy | valid_count for microbatch weighting


@dataclass
class CVConfig:
    """Section 21: deterministic grouped K-fold."""
    type: str = "group"              # group | random
    folds: int = 5
    group_column: str = "report_hash"  # report_hash | patient_id
    seed: int = 2026


@dataclass
class ModelConfig:
    architecture: str = "cls"  # cls baseline | patch_query
    aggregation: str = "transformer"  # transformer | attention_mil (CLS control)
    amp_dtype: str = "auto"  # auto | float16 | bfloat16
    encode_chunk_size: int = 32
    checkpoint_encoder: bool = True
    spatial_layers: int = 1
    global_residual: bool = True
    # section 9: adjacent-slice 2.5D channels (i-1, i, i+1); edge-replicated
    slice_context: int = 3
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    d_model: int = 512
    # section 12: structural embeddings for every DINO feature token
    positional: str = "fourier"      # fourier | sinusoidal | mlp
    fourier_bands: int = 16
    # section 13: within-series transformer over the N slice tokens
    series_transformer: SeriesTransformerConfig = field(
        default_factory=SeriesTransformerConfig)
    # section 14: what the pathology decoder sees
    study_tokens: StudyTokensConfig = field(default_factory=StudyTokensConfig)
    # section 15: the 12 pathology-specific learned queries
    pathology_decoder: PathologyDecoderConfig = field(
        default_factory=PathologyDecoderConfig)
    ff_mult: int = 4
    n_pathologies: int = 12
    independent_heads: bool = False  # preserve existing checkpoint layout
    embedding_init_std: float | None = None


@dataclass
class PreprocessPipelineConfig:
    seed: int = 2026
    study_root: str = "data/train_series"
    series_table: str = "data/train_series.csv"
    meta_cache: str = "data/meta/slice_meta.parquet"
    output_dir: str = "data/processed/train"
    blacklist: list[str] = field(default_factory=list)
    slots: SlotsConfig = field(default_factory=SlotsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    laterality: LateralityConfig = field(default_factory=LateralityConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cv: CVConfig = field(default_factory=CVConfig)


def _selection_from(raw: Any) -> list[SelectionComponent]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [SelectionComponent(**entry) for entry in raw]
    return [
        SelectionComponent(name=name, weight=body["weight"],
                           params={k: v for k, v in body.items() if k != "weight"})
        for name, body in raw.items()
    ]


def _slots_from(raw: Any) -> SlotsConfig:
    if not raw:
        return SlotsConfig()
    defs = {
        slot: [SlotKey(plane=k["plane"], fluid_sensitive=int(k["fluid_sensitive"]),
                       fat_suppression=(None if k.get("fat_suppression") is None
                                        else int(k["fat_suppression"])))
               for k in keys]
        for slot, keys in (raw or {}).get("definitions", {}).items()
    }
    return SlotsConfig(definitions=defs,
                       selection=_selection_from((raw or {}).get("selection")),
                       k_series_per_slot=int((raw or {}).get("k_series_per_slot", 1)))


def load_config(path: str | None = None) -> PreprocessPipelineConfig:
    """Load YAML config into dataclasses. Relative data paths resolve to repo root.

    `preprocessing.normalize_laterality` is honored as an alias for
    `laterality.normalize` (the spec's original key name); an explicit
    `laterality.normalize` wins when both are present.
    """
    if path is None:
        path = os.path.join(REPO, "configs", "preprocess.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> PreprocessPipelineConfig:
    laterality_raw = dict(raw.get("laterality", {}))
    pre_raw = dict(raw.get("preprocessing", {}))
    if "normalize" not in laterality_raw and "normalize_laterality" in pre_raw:
        laterality_raw["normalize"] = bool(pre_raw["normalize_laterality"])
    pre_raw.pop("normalize_laterality", None)
    cfg = PreprocessPipelineConfig(
        seed=raw.get("seed", 2026),
        study_root=raw.get("study_root", "data/train_series"),
        series_table=raw.get("series_table", "data/train_series.csv"),
        meta_cache=raw.get("meta_cache", "data/meta/slice_meta.parquet"),
        output_dir=raw.get("output_dir", "data/processed/train"),
        blacklist=list(raw.get("blacklist", [])),
        slots=_slots_from(raw.get("slots")),
        geometry=GeometryConfig(**raw.get("geometry", {})),
        laterality=LateralityConfig(**laterality_raw),
        preprocessing=PreprocessConfig(**pre_raw),
        sampling=SamplingConfig(**raw.get("sampling", {})),
        output=OutputConfig(**raw.get("output", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        model=_model_from(raw.get("model")),
        loss=LossConfig(**raw.get("loss", {})),
        cv=CVConfig(**raw.get("cv", {})),
    )
    return cfg


def _model_from(raw: Any) -> ModelConfig:
    raw = dict(raw or {})
    enc = EncoderConfig(**raw.pop("encoder", {}))
    st = SeriesTransformerConfig(**raw.pop("series_transformer", {}))
    stt = StudyTokensConfig(**raw.pop("study_tokens", {}))
    pd_cfg = PathologyDecoderConfig(**raw.pop("pathology_decoder", {}))
    return ModelConfig(encoder=enc, series_transformer=st, study_tokens=stt,
                       pathology_decoder=pd_cfg, **raw)


def abspath(cfg_path: str) -> str:
    """Resolve a config-relative data path against the repo root."""
    return cfg_path if os.path.isabs(cfg_path) else os.path.join(REPO, cfg_path)
