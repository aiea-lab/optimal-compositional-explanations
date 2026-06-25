"""Utilities for activation loading, unit selection, and bitmap construction.

Supports preprocessing steps used by all explanation methods.
"""

import random
from typing import List, Tuple

import torch
import sklearn.cluster as scikit_cluster

from . import vecquantile
from . import constants as C
from src.model_wrapper import Place365Model, DenseNetPlace365


def get_layer_activations(cfg):
    """Returns the activations for the given layer and dataset.
    Args:
        cfg (Config): configuration object containing the model, dataset and layer information

    Returns:
        activations (torch.Tensor): activations for the given layer and dataset
    """

    weights = cfg.get_weights()  # Get path to weights
    dataset_name = cfg.get_dataset_name()
    layer_name = cfg.get_layer()
    activation_dir = (
        cfg.get_root_activations() + f"/{dataset_name}/{cfg.get_model()}"
    )  # Dir where to store activations
    if cfg.get_model() == "densenet161":
        # Densenet161 needs a custom loading
        model_wrapper = DenseNetPlace365(
            model_name=cfg.get_model(), weights=weights, device=cfg.get_device()
        )
    else:
        model_wrapper = Place365Model(
            model_name=cfg.get_model(), weights=weights, device=cfg.get_device()
        )

    # Get Masks Information from the concept dataset
    if dataset_name == "broden":
        model_wrapper.set_loader(dataset_name, cfg)
    else:
        model_wrapper.set_loader(dataset_name)

    layer_activations = model_wrapper.get_layer_activations(layer_name, activation_dir)
    return layer_activations


def get_selected_units(layer_activations, units=None, random_units=0):
    """Returns the selected units for the given layer activations.
    Args:
        layer_activations (torch.Tensor): activations for the given layer and dataset
        units (List[int], optional): list of units to select. If None, all units are selected. Defaults to None.
        random_units (int, optional): number of random units to select. If 0, no random units are selected. Defaults to 0.
    Returns:
        selected_units (List[int]): list of selected units for the given layer activations
    """
    if units is not None:
        for unit in units:
            if int(unit) >= layer_activations.shape[1]:
                raise ValueError(f"Unit {unit} is out of range")
        selected_units = [int(unit) for unit in units]
    elif random_units > 0:
        selected_units = random.sample(range(layer_activations.shape[1]), random_units)
    else:
        selected_units = range(layer_activations.shape[1])
    return selected_units


def compute_activation_ranges(
    activations: torch.Tensor, num_clusters: int
) -> List[Tuple]:
    """Compute activation ranges for each unit.

    Args:
        activations (torch.Tensor): Activations of the unit.
        num_clusters (int): Number of clusters.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each unit.
    """
    if num_clusters == 1:
        # Case vanilla compositional and netdissect range
        # Avoid zero is set to false like in the compositional paper
        threshold = quantile_threshold(
            activations, quantile=C.NETDISSECT_QUANTILE, avoid_zero=False
        )
        activation_ranges = [(threshold, torch.tensor(float("inf")))]
    else:
        activations = activations.reshape(-1, 1)
        # Remove zeros from activations if there is a relu activation
        if torch.all(activations >= 0):
            activations = activations[activations > 0]
            activations = activations.reshape(-1, 1)
        # Compute activation ranges
        clusters = scikit_cluster.KMeans(n_clusters=num_clusters, random_state=0).fit(
            activations
        )
        activation_ranges = build_ranges_from_clusters(
            activations, clusters.labels_, num_clusters
        )
    return activation_ranges


def build_ranges_from_clusters(
    activations: torch.Tensor, clusters: List[int], num_clusters: int
) -> List[tuple]:
    """Build activation ranges from clusters.

    Args:
        activations (torch.Tensor): Activations of the unit.
        clusters (List[int]): Clusters indexes of the activations.
        num_clusters (int): Number of clusters.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each cluster.
    """

    activations_ranges = []
    for label in range(num_clusters):
        cluster_activations = activations[clusters == label]
        lower_bound = torch.min(cluster_activations)
        upper_bound = torch.max(cluster_activations)
        activations_ranges.append((lower_bound.item(), upper_bound.item()))
    return activations_ranges


def quantile_threshold(
    layer_activations: torch.Tensor,
    quantile: float,
    *,
    avoid_zero: bool,
    batch_size=64,
    seed=1,
) -> torch.Tensor:
    """
    Determine thresholds for neuron activations for each neuron.

    Args:
        layer_activations (torch.Tensor): Activations of the layer.
        quantile (float): Quantile to use.
        avoid_zero (bool): Whether to remove zeros from the activations.
        batch_size (int): Batch size to use.
        seed (int): Seed to use for the quantile vector.

    Returns:
        thresholds (torch.Tensor): Thresholds for each neuron.
    """
    quant = vecquantile.QuantileVector(depth=1, seed=seed)
    for i in range(0, layer_activations.shape[0], batch_size):
        batch = layer_activations[i : i + batch_size]
        batch = batch.flatten().reshape(-1, 1)
        if avoid_zero:
            batch = batch[batch != 0].reshape(-1, 1)
        quant.add(batch)
    thresholds = quant.readout(1000)[:, int(1000 * (1 - quantile) - 1)]
    return torch.tensor(thresholds)


def compute_bitmaps(
    activations: torch.Tensor, activation_range: Tuple, mask_shape: List[int]
) -> torch.Tensor:
    """Get the bitmaps of the unit.

    This function upsamples the activations to the original size of the
    image and then binarize them.
    Args:
        activations (torch.Tensor): Activations of the unit.
        activation_range (Tuple): Activation range of the unit.
        mask_shape (List[int]): Shape of the mask.

    Returns:
        bitmaps (torch.Tensor): Bitmaps of the unit.
    """
    lower, upper = activation_range
    upsampled_activations = torch.nn.functional.interpolate(
        activations.unsqueeze(1), size=mask_shape, mode="bilinear"
    )
    upsampled_activations = upsampled_activations.squeeze(1)
    bitmaps = torch.where(
        (upsampled_activations > lower) & (upsampled_activations < upper), True, False
    )
    bitmaps = bitmaps.reshape(bitmaps.shape[0], -1)
    return bitmaps
