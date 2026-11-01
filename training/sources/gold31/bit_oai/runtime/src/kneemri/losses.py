"""Losses for sections 18 (weighted soft-label BCE) and 19 (pairwise AUC
ranking loss).

L_total = L_soft_bce + lambda_rank * L_rank
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftBCELoss(nn.Module):
    """Section 18: confidence-weighted soft-label BCE.

        L = sum(conf_ik * BCEWithLogits(logit_ik, target_ik)) / sum(conf_ik)

    conf_ik = 0 for missing/unknown labels (masked out entirely); otherwise
    gold_weight for expert-labeled (gold) studies, pseudo_default_weight for
    pseudo-labeled ones. All weights come from config.
    """

    def __init__(self, gold_weight: float = 1.0,
                 pseudo_default_weight: float = 0.7, normalization: str = 'weight_sum'):
        super().__init__()
        self.gold_weight = gold_weight
        self.pseudo_default_weight = pseudo_default_weight
        if normalization not in ('weight_sum', 'valid_count'):
            raise ValueError('unknown loss normalization')
        if min(gold_weight, pseudo_default_weight) < 0:
            raise ValueError('loss weights must be nonnegative')
        self.normalization = normalization

    def denominator(self, targets, label_mask, is_gold):
        mask = label_mask.float() * torch.isfinite(targets)
        if self.normalization == 'valid_count':
            return mask.sum()
        gold = is_gold.unsqueeze(1) if is_gold.ndim == 1 else is_gold
        return (mask * torch.where(gold, self.gold_weight, self.pseudo_default_weight)).sum()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                label_mask: torch.Tensor, is_gold: torch.Tensor) -> torch.Tensor:
        """logits/targets/label_mask: (B, 12); is_gold: (B,) bool.
        NaN targets are treated as missing (mask forced to 0)."""
        label_mask = label_mask.float()
        label_mask = label_mask * torch.isfinite(targets).float()
        safe_targets = torch.nan_to_num(targets, nan=0.0).clamp(0.0, 1.0)
        gold_mask = is_gold.unsqueeze(1) if is_gold.ndim == 1 else is_gold
        if gold_mask.shape not in ((len(logits),1), logits.shape):
            raise ValueError('gold mask must be study-level or target-level')
        w = torch.where(gold_mask,
                        torch.full_like(label_mask, self.gold_weight),
                        torch.full_like(label_mask, self.pseudo_default_weight))
        w = w * label_mask
        if w.sum() <= 0:
            return logits.sum() * 0.0     # differentiable zero, no NaN
        loss = F.binary_cross_entropy_with_logits(
            logits, safe_targets, reduction="none")
        denominator = w.sum() if self.normalization == 'weight_sum' else label_mask.sum().clamp(min=1)
        return (loss * w).sum() / denominator


class PairwiseAucRankLoss(nn.Module):
    """Section 19: the metric is per-pathology ROC-AUC, so push
    s_pos > s_neg with softplus(-(s_pos - s_neg)).

    Pairs are formed within one pathology across the batch. Pathologies with
    no positive or no negative in the batch are SKIPPED (never NaN). Cost is
    O(P*G) per pathology with P+G <= B — small by construction, no large
    O(B^2) matrix over pathologies.
    """

    def __init__(self, positive_threshold: float = 0.5):
        super().__init__()
        self.positive_threshold = positive_threshold

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                label_mask: torch.Tensor) -> torch.Tensor:
        label_mask = label_mask.float() * torch.isfinite(targets).float()
        targets = torch.nan_to_num(targets, nan=0.0)
        total, terms = logits.new_zeros(()), 0
        B = logits.shape[0]
        for k in range(logits.shape[1]):
            m = label_mask[:, k] > 0
            y = targets[:, k]
            pos = m & (y > self.positive_threshold)
            neg = m & (y <= self.positive_threshold)
            n_pos, n_neg = int(pos.sum()), int(neg.sum())
            if n_pos == 0 or n_neg == 0:
                continue
            s = logits[:, k]
            # (n_pos, n_neg) pairwise margin matrix — bounded by B^2/4
            diff = s[pos].unsqueeze(1) - s[neg].unsqueeze(0)
            total = total + F.softplus(-diff).mean()
            terms += 1
        if terms == 0:
            return logits.sum() * 0.0
        return total / terms


class CombinedLoss(nn.Module):
    """L_total = L_soft_bce + lambda_rank * L_rank (section 19)."""

    def __init__(self, gold_weight: float = 1.0,
                 pseudo_default_weight: float = 0.7,
                 lambda_rank: float = 0.1, normalization: str = 'weight_sum'):
        super().__init__()
        self.soft_bce = SoftBCELoss(gold_weight, pseudo_default_weight, normalization)
        self.rank = PairwiseAucRankLoss()
        self.lambda_rank = lambda_rank

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                label_mask: torch.Tensor, is_gold: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        l_bce = self.soft_bce(logits, targets, label_mask, is_gold)
        l_rank = self.rank(logits, targets, label_mask) if self.lambda_rank else logits.sum() * 0.0
        total = l_bce + self.lambda_rank * l_rank
        return total, {"soft_bce": l_bce.detach(), "rank": l_rank.detach()}
