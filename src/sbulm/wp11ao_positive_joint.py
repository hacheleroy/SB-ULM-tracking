"""Nonnegative four-slot alive/count-by-clutter joint for crowded references.

The one-slot factors are shared with the squarefree computer. Ordered
two-slot coefficients are formed by explicit disjoint pairs, then two
pairs are joined by positive subset zeta sums over non-target bits.
No Mobius inversion or alive-count Fourier inversion is used.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .wp11v_tempering_mode_audit import _ExactComputer


def _ordered_pairs(computer: _ExactComputer, beta: float,
                   dtype: type) -> tuple[np.ndarray, float, float]:
    started = time.process_time()
    scores = computer.base + beta * computer.gp
    logz1 = float(logsumexp(scores))
    weights = np.exp(scores.astype(dtype) - dtype(logz1))
    masks = computer.subsets.astype(np.int32)
    lengths = computer.lengths
    pair = np.zeros((2 * computer.model.sequence.n_frames + 1, 1 << computer.n),
                    dtype=dtype)
    for mask, length, weight in zip(masks, lengths, weights, strict=True):
        disjoint = (masks & mask) == 0
        np.add.at(pair, (lengths[disjoint] + length, masks[disjoint] | mask),
                  weight * weights[disjoint])
    return pair, logz1, time.process_time() - started


def _layout(computer: _ExactComputer, pair: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    offsets = np.cumsum((0,) + computer.cards[:-1])
    frame_bits = tuple(tuple(range(offsets[f], offsets[f] + computer.cards[f]))
                       for f in computer.frames)
    target_bits = frame_bits[0] + frame_bits[1]
    other_bits = tuple(bit for bit in range(computer.n) if bit not in target_bits)
    other = np.arange(1 << len(other_bits), dtype=np.int32)
    expanded_other = np.zeros(len(other), dtype=np.int32)
    for compact, original in enumerate(other_bits):
        expanded_other |= ((other >> compact) & 1) << original
    target_masks = np.arange(1 << len(target_bits), dtype=np.int32)
    expanded_target = np.zeros(len(target_masks), dtype=np.int32)
    for compact, original in enumerate(target_bits):
        expanded_target |= ((target_masks >> compact) & 1) << original
    split = np.empty((len(target_masks), pair.shape[0], len(other)), dtype=pair.dtype)
    for target, expanded in enumerate(expanded_target):
        split[target] = pair[:, expanded_other | expanded]
    return split, tuple(len(bits) for bits in frame_bits)


def positive_four_slot_joint(computer: _ExactComputer, beta: float,
                             *, extended: bool = True) -> dict[str, Any]:
    """Sum the ordered four-slot joint with nonnegative arithmetic.

    p[a,S] is the ordered two-slot weight at alive length a and detection
    union S. Split S into bits in the two clutter frames and all other
    bits. For each disjoint target-frame mask pair (H,R), zeta-sum
    p[b,R,T] over T contained in the complement of the first pair's
    other-frame mask. Dot products then sum positive contributions by
    total alive length and the two observed clutter counts.
    """
    if computer.model.n_slots != 4 or not 0 <= beta <= 1:
        raise ValueError("requires four slots and beta in [0,1]")
    started = time.process_time()
    dtype = np.longdouble if extended else np.float64
    pair, logz1, pair_seconds = _ordered_pairs(computer, beta, dtype)
    split, (first_bits, second_bits) = _layout(computer, pair)
    del pair
    n_other = split.shape[-1]
    complement = n_other - 1 - np.arange(n_other)
    alive_pairs = split.shape[1]
    index_sum = np.add.outer(np.arange(alive_pairs), np.arange(alive_pairs)).ravel()
    mass = np.zeros((computer.n_alive,) + computer.shape, dtype=dtype)
    for second_target in range(split.shape[0]):
        zeta = split[second_target].copy()
        for bit in range(int(np.log2(n_other))):
            view = zeta.reshape(alive_pairs, -1, 2, 1 << bit)
            view[:, :, 1, :] += view[:, :, 0, :]
        opposite = np.ascontiguousarray(zeta[:, complement].T)
        for first_target in range(split.shape[0]):
            if first_target & second_target:
                continue
            first_clutter = first_bits - int((first_target | second_target)
                                             & ((1 << first_bits) - 1)).bit_count()
            second_clutter = second_bits - int((first_target | second_target)
                                               >> first_bits).bit_count()
            sums = split[first_target] @ opposite
            np.add.at(mass[:, first_clutter, second_clutter],
                      index_sum, sums.ravel())
    if np.any(mass < 0):
        raise ValueError("positive joint produced a negative cell")
    total = mass.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("nonpositive or nonfinite four-slot mass")
    model = computer.model
    logz = (dtype(computer.n) * np.log(dtype(model.clutter_intensity))
            - dtype(model.sequence.n_frames)
            * dtype(model.expected_clutter_per_frame)
            + 4 * dtype(logz1) + np.log(total))
    joint = np.asarray(mass / total, dtype=float)
    return {"beta": float(beta), "joint": joint.tolist(),
            "log_partition": float(logz),
            "normalized_four_slot_mass": float(total),
            "minimum_cell": float(joint.min()),
            "precision": "longdouble" if extended else "float64",
            "pair_cpu_seconds": pair_seconds,
            "cpu_seconds": time.process_time() - started}
