"""High-level utilities for compositional explanation pipelines.

Coordinates search-method dispatch, execution, and result persistence.
"""

import os
import pickle
from timeit import default_timer as timer

import torch
from tqdm import tqdm

from . import activations_utils
from src import algorithms
from compositional import formula as F
from compositional import mmesh, vanilla_search, beam_optimal, optimal


def get_explanations(
    segmentations,
    activation_masks,
    *,
    heuristic="mmesh",
    config=None,
    segmentations_info=None,
    disjoint_info=None,
    length=3,
):
    """Compute the heuristic score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        segmentations (Iterable): An iterable of segmentation masks. Each mask is a tensor of shape (N, H*W).
        activation_masks (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        heuristic (str): The heuristic to use. Available heuristics: mmesh, optimal, beam_optimal, none.
        config (Settings): The configuration object containing the parameters for the search.
        segmentations_info (tuple): A tuple containing the pre-computed information for the segmentations. The content of the tuple depends on the heuristic used.
        disjoint_info (list of list): A list of lists containing the disjoint information for each pair of concepts. Each element is a list of booleans indicating whether the two concepts are disjoint or not.
        length (int): The maximum length of the formulas to consider.

    Returns:
        best_label (int): The label of the best concept.
        best_iou (float): The IOU of the best concept.
        visited (int): The number of visited nodes.
        expanded (int): The number of expanded nodes.
        estimated (int): The number of estimated nodes.
    """

    if segmentations_info is None and heuristic != "none":
        raise ValueError(
            "segmentations_info must be provided when heuristic is not none"
        )
    # Compute commong parameters
    if length == 1:
        # vanilla netdissect explanation
        return algorithms.get_netdissect_explanation(
            activation_masks, segmentations
        ) + (
            len(segmentations),
            0,
            0,
        )  # It computes the IoU for all the concepts
    if heuristic == "mmesh":
        return mmesh.compute_mmesh_explanations(
            bitmaps=activation_masks,
            masks=segmentations,
            masks_info=segmentations_info,
            config=config,
        )
    elif heuristic == "none":
        return vanilla_search.compute_vanilla_explanations(
            bitmaps=activation_masks, masks=segmentations, config=config
        )
    elif heuristic == "beam_optimal":
        masks_info = segmentations_info[
            2
        ]  # Beam optimal uses only the quantities for the heuristic, so we extract them from the segmentations info
        return beam_optimal.compute_beam_optimal_explanations(
            bitmaps=activation_masks,
            masks=segmentations,
            masks_info=masks_info,
            disjoint_info=disjoint_info,
            config=config,
        )
    elif heuristic == "optimal":
        masks_info = segmentations_info[
            2
        ]  # Optimal uses only the quantities for the heuristic, so we extract them from the segmentations info
        return optimal.compute_optimal_explanations(
            bitmaps=activation_masks,
            masks=segmentations,
            masks_info=masks_info,
            disjoint_info=disjoint_info,
            config=config,
        )

    else:
        raise ValueError(
            f"Unknown heuristic {heuristic}. "
            "Available heuristics: mmesh, optimal, none."
        )


def compute_compositional_explanations(
    *,
    masks,
    masks_labels,
    masks_info,
    disjoint_info,
    activations,
    units,
    config,
    verbose=True,
):
    """Compute the compositional explanations for the given activations and masks.
    Args:
        masks (list): A list of segmentation masks. Each mask is a tensor of shape (N, H*W).
        masks_labels (list): A list of labels for the segmentation masks.
        masks_info (tuple): A tuple containing the pre-computed information for the segmentations.
        disjoint_info (list of list): A list of lists containing the disjoint information for each pair of concepts. Each element is a list of booleans indicating whether the two concepts are disjoint or not.
        activations (torch.Tensor): A tensor of shape (N, C, H, W) containing the activations of the layer.
        units (list): A list of unit indices to compute the explanations for.
        config (Settings): The configuration object containing the parameters for the search.
        verbose (bool): Whether to print the results for each unit or not.
    """
    num_clusters = config.get_num_clusters()
    heuristic = config.get_heuristic()
    length = config.get_length()
    device = config.get_device()
    mask_shape = config.get_mask_shape()
    results_dir = config.get_result_dir()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for unit in tqdm(units, desc="Computing Compostional explanations per unit"):
        unit_activations = activations[:, unit, :, :]

        if torch.count_nonzero(unit_activations) < num_clusters:
            print(
                f"Unit {unit} has very low activation {torch.count_nonzero(unit_activations)}, skipping it"
            )
            continue

        # Compute activation range to be kept in the masks
        activation_ranges = activations_utils.compute_activation_ranges(
            unit_activations, num_clusters
        )
        for cluster_index, activation_range in enumerate(sorted(activation_ranges)):

            # CHANGEME: We currently skip the optimal explanations for all the clusters except the last one since it can be very slow to compute. If you want to compute the optimal explanations for all the clusters, please manually edit or remove this part of the code.
            if (
                config.get_dataset_name() == "broden"
                and heuristic == "optimal"
                and num_clusters > 1
            ):
                if cluster_index != num_clusters - 1:
                    print(
                        f"Running optimal explanations for lower unspecialized cluster {cluster_index} in Broden can be too slow"
                        f" therefore, this script automatically skips all the clusters exept the last one."
                        f" If you want to run the optimal explanations for all the clusters, please manually edit the compositional_utils file searching for the instruction CHANGEME."
                    )
                    continue
            # END CHANGEME

            file_algo_results = (
                f"{results_dir}/unit_{unit}_cluster_{cluster_index}_activation_{activation_range}_"
                + f"{length}.pickle"
            )
            if os.path.exists(file_algo_results):
                with open(file_algo_results, "rb") as file:
                    loaded_info = pickle.load(file)
                    (
                        best_label,
                        string_label,
                        best_iou,
                        visited,
                        expanded,
                        estimated,
                        time_taken,
                    ) = loaded_info

                    if string_label != F.get_formula_str(best_label, masks_labels):
                        raise ValueError(
                            f"The mapping used during the computation of explanation is different from the one used during the collection. String label does not match the formula {string_label} - {F.get_formula_str(best_label, masks_labels)}"
                        )

            else:
                print(
                    f"Computing {heuristic} explanation for Unit {unit} Cluster {cluster_index}."
                )

                # Compute binary masks
                bitmaps = activations_utils.compute_bitmaps(
                    unit_activations,
                    activation_range,
                    mask_shape=mask_shape,
                )
                bitmaps = bitmaps.to(device)
                start_time = timer()
                best_label, best_iou, visited, expanded, estimated = get_explanations(
                    masks,
                    bitmaps,
                    heuristic=heuristic,
                    config=config,
                    segmentations_info=masks_info,
                    disjoint_info=disjoint_info,
                    length=length,
                )
                end_time = timer()
                time_taken = end_time - start_time
                string_label = F.get_formula_str(best_label, masks_labels)
                with open(file_algo_results, "wb") as file:
                    pickle.dump(
                        (
                            best_label,
                            string_label,
                            best_iou,
                            visited,
                            expanded,
                            estimated,
                            time_taken,
                        ),
                        file,
                    )

            if verbose:
                print(
                    f"Unit: {unit} - "
                    + f"Cluster: {cluster_index} - "
                    + f"Best Label: {string_label} - "
                    + f"Number Labels: {best_label} - "
                    + f"Best IoU: {round(best_iou,4)} - "
                    + f"Visited: {visited} - "
                    + f"Expanded: {expanded} - "
                    + f"Estimated: {estimated}"
                    + f" - Time: {time_taken:.2f} seconds"
                )


def load_compositional_explanations(*, activations, units, config):
    """Load the compositional explanations for the given activations and units.
    Args:
        activations (torch.Tensor): A tensor of shape (N, C, H, W) containing the activations of the layer.
        units (list): A list of unit indices to compute the explanations for.
        config (Settings): The configuration object containing the parameters for the search.
    Returns:
        list: A list of tuples containing the loaded explanations for each unit.
    """
    num_clusters = config.get_num_clusters()
    length = config.get_length()
    results_dir = config.get_result_dir()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results = []
    not_found = []
    for unit in tqdm(units, desc="Loading Compostional explanations"):
        unit_activations = activations[:, unit, :, :]

        if torch.count_nonzero(unit_activations) < num_clusters:
            print(
                f"Unit {unit} has very low activation {torch.count_nonzero(unit_activations)}, skipping it"
            )
            continue

        # Compute activation range to load the proper file
        activation_ranges = activations_utils.compute_activation_ranges(
            unit_activations, num_clusters
        )
        for cluster_index, activation_range in enumerate(sorted(activation_ranges)):

            file_algo_results = (
                f"{results_dir}/unit_{unit}_cluster_{cluster_index}_activation_{activation_range}_"
                + f"{length}.pickle"
            )
            if os.path.exists(file_algo_results):
                with open(file_algo_results, "rb") as file:
                    loaded_info = pickle.load(file)
                    (
                        best_label,
                        string_label,
                        best_iou,
                        visited,
                        expanded,
                        estimated,
                        time_taken,
                    ) = loaded_info
            else:
                (
                    best_label,
                    string_label,
                    best_iou,
                    visited,
                    expanded,
                    estimated,
                    time_taken,
                ) = (None, None, None, None, None, None, None)
                not_found.append((unit, cluster_index, activation_range))

            results.append(
                (
                    unit,
                    cluster_index,
                    activation_range,
                    best_label,
                    string_label,
                    best_iou,
                    visited,
                    expanded,
                    estimated,
                    time_taken,
                )
            )
    print(f"Units with missing explanations: {len(not_found)} over {len(results)}")
    return results
