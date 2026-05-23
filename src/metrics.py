"""Metric utilities for binary mask evaluation.

Includes IoU and related low-level scoring primitives.
"""

import functools

import torch


@functools.lru_cache(maxsize=10)
def compute_hits(vector):
    """Compute the number of ones in the given vector.
    Args:
        vector (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        hits (int): The number of ones in the given vector.
    """
    return torch.count_nonzero(vector)


def iou(vector1, vector2):
    """Compute the intersection over union between two vectors.
    Args:
        vector1 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        vector2 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        iou (float): The intersection over union between the two vectors.
    """
    intersection = torch.count_nonzero(vector1 & vector2).item()
    v1_size = compute_hits(vector1).item()
    v2_size = compute_hits(vector2).item()
    score = (
        intersection / (v1_size + v2_size - intersection)
        if v1_size + v2_size - intersection > 0
        else 0
    )
    return score
