"""Path-level heuristic utilities for estimating IoU bounds during search."""

from compositional import optimal_sample_heuristic, optimal_sum_heuristic
from utils import optimal_utils
from utils.constants import (
    INDEX_NODE_IOU_ESTI,
    INDEX_NODE_OPS,
    INDEX_OR,
    INDEX_AND,
    INDEX_NOT,
)


def is_in(op, list_of_lists):
    """
    Check if the operation is in any of the lists in list_of_lists.

    Args:
        op (str): The operation to check.
        list_of_lists (list of list): The lists to check against.

    Returns:
        bool: True if the operation is found in any of the lists, False otherwise.
    """
    return any(op in lst for lst in list_of_lists)


def max_min_iou_from_union_intersection(
    max_min_intersection, max_min_union, minimum_threshold=0.0
):
    """Compute the maximum and minimum IoU from the maximum and minimum intersection and union.

    Args:
        max_min_intersection (tuple): A tuple containing the maximum and minimum intersection.
        max_min_union (tuple): A tuple containing the maximum and minimum union.
        minimum_threshold (float): The minimum threshold for the IoU. If the maximum IoU is below this threshold, both maximum and minimum IoU will be set to 0.

    Returns:
        tuple: A tuple containing the maximum and minimum IoU.
    """
    max_intersection, min_intersection = max_min_intersection
    max_union, min_union = max_min_union
    max_iou = max_intersection / min_union if min_union > 0 else 0.0

    if max_iou < minimum_threshold:
        return 0.0, 0.0

    min_iou = min_intersection / max_union if max_union > 0 else 0.0
    return max_iou, min_iou


def get_combo_quantities(
    max_min_intersection_a, max_min_union_a, max_min_intersection_b, max_min_union_b
):
    """Get the combined quantities for two paths. This is used for the AND-OR and AND-AND NOT chains.

    Args:
        max_min_intersection_a (tuple): A tuple containing the maximum and minimum intersection for path A.
        max_min_union_a (tuple): A tuple containing the maximum and minimum union for path A.
        max_min_intersection_b (tuple): A tuple containing the maximum and minimum intersection for path B.
        max_min_union_b (tuple): A tuple containing the maximum and minimum union for path B.
    Returns:
        dict: A dictionary containing the combined quantities.
    """
    (max_intersection_a, min_intersection_a), (max_union_a, min_union_a) = (
        max_min_intersection_a,
        max_min_union_a,
    )
    (max_intersection_b, min_intersection_b), (max_union_b, min_union_b) = (
        max_min_intersection_b,
        max_min_union_b,
    )
    max_intersection = max(max_intersection_a, max_intersection_b)
    min_intersection = min(min_intersection_a, min_intersection_b)
    max_union = max(max_union_a, max_union_b)
    min_union = min(min_union_a, min_union_b)
    return (max_intersection, min_intersection), (max_union, min_union)


def estimate_pairwise_path(
    max_min_intersection_a,
    max_min_union_a,
    max_min_intersection_b,
    max_min_union_b,
    minimum_threshold=0.0,
):
    """Estimate the maximum and minimum IoU for a combined path (e.g., AND-OR or OR-NOT) based on the maximum and minimum intersection and union of the individual paths.
    Args:
        max_min_intersection_a (tuple): A tuple containing the maximum and minimum intersection for path A.
        max_min_union_a (tuple): A tuple containing the maximum and minimum union for path A.
        max_min_intersection_b (tuple): A tuple containing the maximum and minimum intersection for path B.
        max_min_union_b (tuple): A tuple containing the maximum and minimum union for path B.
        minimum_threshold (float): The minimum threshold for the IoU. If the maximum IoU is below this threshold, both maximum and minimum IoU will be set to 0.

    Returns:
        tuple: A tuple containing the maximum and minimum IoU for the combined path.
    """
    max_min_intersection, max_min_union = get_combo_quantities(
        max_min_intersection_a, max_min_union_a, max_min_intersection_b, max_min_union_b
    )
    max_iou, min_iou = max_min_iou_from_union_intersection(
        max_min_intersection, max_min_union, minimum_threshold
    )
    return max_iou, min_iou


def update_paths_iou(
    *,
    heuristic_name,
    node,
    label_mapping,
    heuristic_info,
    max_improvement,
    disjoint_info,
    num_hits,
    max_size_mask,
    max_length,
    minimum_threshold=0.0,
):
    """Update the IoU estimation of the paths for a given node. This is used to update the IoU estimation from sum to sample estimation.

    Args:
        node (list): The node to update the paths for.
        label_mapping (dict): A dictionary mapping labels to their quantities.
        heuristic_info (tuple): A tuple containing the information to compute the heuristic.
        max_improvement (float): The maximum improvement allowed for the paths. This is used to limit the search space of the paths.
        disjoint_info (dict): A dictionary containing the disjoint information for the labels. This is used to discard labels that are disjoint with the current label.
        num_hits (int): The number of hits to consider for the estimation. This is used to limit the search space of the paths.
        max_size_mask (int): The maximum size of the mask to consider for the estimation. This is used to limit the search space of the paths.
        max_length (int): The maximum length of the paths. This is used to limit the search space of the paths.
        minimum_threshold (float): The minimum threshold for the IoU. If the maximum IoU is below this threshold, both maximum and minimum IoU will be set to 0.

    Returns:
        tuple: A tuple containing the new maximum and minimum IoU for the paths.
    """
    if heuristic_name == "sum":
        heuristic = optimal_sum_heuristic
    elif heuristic_name == "sample":
        heuristic = optimal_sample_heuristic
    else:
        raise ValueError(f"Unknown heuristic name: {heuristic_name}")
    _, neuron_quantities, _ = heuristic_info

    label = node[2]  # Starting label
    previous_paths_to_expand = node[3]

    if previous_paths_to_expand is not None:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = (
            previous_paths_to_expand
        )
    else:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = (
            [],
            [],
            [],
            [],
        )

    label_quantities = optimal_utils.estimate_label_quantities(
        heuristic,
        label=label,
        label_mapping=label_mapping,
        heuristic_info=heuristic_info,
        max_size_mask=max_size_mask,
        disjoint_info=disjoint_info,
    )

    if label_quantities is None:
        # Label discarded at the previous step
        return 0.0, 0.0

    or_condition = is_in("OR", previous_or_paths)
    and_condition = is_in("AND", previous_and_paths)
    and_not_condition = is_in("NOT", previous_and_not_paths)
    and_and_not_condition = is_in(["AND", "NOT"], previous_and_paths) or is_in(
        ["AND", "NOT"], previous_and_not_paths
    )
    and_or_condition = is_in(["OR", "AND"], previous_or_paths) or is_in(
        ["OR", "AND"], previous_and_paths
    )
    or_not_condition = is_in(["OR", "NOT"], previous_or_paths) or is_in(
        ["OR", "NOT"], previous_and_not_paths
    )
    every_condition = (
        is_in(["OR", "AND", "NOT"], previous_or_paths)
        or is_in(["OR", "AND", "NOT"], previous_and_paths)
        or is_in(["OR", "AND", "NOT"], previous_and_not_paths)
    )

    # Individual estimation
    ind_new_max, ind_new_min = optimal_utils.estimate_min_max_iou_from_label_info(
        label_quantities=label_quantities,
        neuron_quantities=neuron_quantities,
        num_hits=num_hits,
        minimum_threshold=minimum_threshold,
    )

    # Exclusive paths update
    if or_condition:
        or_max_min_intersection, or_max_min_union = heuristic.or_chain_estimation(
            label,
            label_quantities=label_quantities,
            neuron_quantities=neuron_quantities,
            max_improvement=max_improvement,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
            max_length=max_length,
        )
        or_new_max, or_new_min = max_min_iou_from_union_intersection(
            or_max_min_intersection, or_max_min_union, minimum_threshold
        )
    else:
        or_new_max, or_new_min = 0, 0
    if and_not_condition:
        and_not_max_min_intersection, and_not_max_min_union = (
            heuristic.and_not_chain_estimation(
                label_quantities=label_quantities,
                neuron_quantities=neuron_quantities,
                max_improvement=max_improvement,
                num_hits=num_hits,
                max_size_mask=max_size_mask,
            )
        )
        and_not_new_max, and_not_new_min = max_min_iou_from_union_intersection(
            and_not_max_min_intersection, and_not_max_min_union, minimum_threshold
        )
    else:
        and_not_new_max, and_not_new_min = 0, 0
    if and_condition:
        # This covers both AND and AND-AND NOT chains
        and_max_min_intersection, and_max_min_union = heuristic.and_chain_estimation(
            label_quantities=label_quantities,
            neuron_quantities=neuron_quantities,
            max_improvement=max_improvement,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
        )
        and_new_max, and_new_min = max_min_iou_from_union_intersection(
            and_max_min_intersection, and_max_min_union, minimum_threshold
        )
    else:
        and_new_max, and_new_min = 0, 0

    # Combined paths update
    if and_and_not_condition:
        and_andnot_max_min_intersection, and_andnot_max_min_union = (
            heuristic.and_chain_estimation(
                label_quantities=label_quantities,
                neuron_quantities=neuron_quantities,
                max_improvement=max_improvement,
                num_hits=num_hits,
                max_size_mask=max_size_mask,
            )
        )
        and_andnot_new_max, and_andnot_new_min = max_min_iou_from_union_intersection(
            and_andnot_max_min_intersection, and_andnot_max_min_union, minimum_threshold
        )
    else:
        and_andnot_new_max, and_andnot_new_min = 0, 0
    if and_or_condition:
        comb_and_or_new_max, comb_and_or_new_min = estimate_pairwise_path(
            and_max_min_intersection,
            and_max_min_union,
            or_max_min_intersection,
            or_max_min_union,
            minimum_threshold=minimum_threshold,
        )
    else:
        comb_and_or_new_max, comb_and_or_new_min = 0, 0
    if or_not_condition:
        or_not_new_max, or_not_new_min = estimate_pairwise_path(
            or_max_min_intersection,
            or_max_min_union,
            and_not_max_min_intersection,
            and_not_max_min_union,
            minimum_threshold=minimum_threshold,
        )
    else:
        or_not_new_max, or_not_new_min = 0, 0

    if every_condition:
        # This covers the everything chain
        and_or_max_min_intersection, and_or_max_min_union = get_combo_quantities(
            and_max_min_intersection,
            and_max_min_union,
            or_max_min_intersection,
            or_max_min_union,
        )
        every_max_min_intersection, every_max_min_union = get_combo_quantities(
            and_or_max_min_intersection,
            and_or_max_min_union,
            and_not_max_min_intersection,
            and_not_max_min_union,
        )
        every_new_max, every_new_min = max_min_iou_from_union_intersection(
            every_max_min_intersection, every_max_min_union, minimum_threshold
        )

    else:
        every_new_max, every_new_min = 0, 0

    new_max = max(
        ind_new_max,
        or_new_max,
        and_not_new_max,
        and_new_max,
        and_andnot_new_max,
        comb_and_or_new_max,
        or_not_new_max,
        every_new_max,
    )
    new_min = min(
        ind_new_min,
        or_new_min,
        and_not_new_min,
        and_new_min,
        and_andnot_new_min,
        comb_and_or_new_min,
        or_not_new_min,
        every_new_min,
    )

    return new_max, new_min


def estimate_paths_iou(
    heuristic_name,
    node,
    label_mapping,
    heuristic_info,
    max_min_improvement,
    *,
    num_hits,
    max_size_mask,
    max_length,
    disjoint_info,
    minimum_threshold=0,
):
    """
    Estimate the IoU of a label using the optimal heuristic.

    Args:
        label (F.Formula): The label to estimate the IoU for.
        heuristic_info (tuple): The information to compute the heuristic.

    Returns:
        float: Estimated IoU of the label.
    """
    if heuristic_name == "sum":
        heuristic = optimal_sum_heuristic
    elif heuristic_name == "sample":
        heuristic = optimal_sample_heuristic
    else:
        raise ValueError(f"Unknown heuristic name: {heuristic_name}")
    _, neuron_quantities, _ = heuristic_info

    label = node[2]  # Starting label
    previous_paths_to_expand = node[3]
    next_path = node[1]

    label_quantities = optimal_utils.estimate_label_quantities(
        heuristic,
        label=label,
        label_mapping=label_mapping,
        heuristic_info=heuristic_info,
        max_size_mask=max_size_mask,
        disjoint_info=disjoint_info,
    )

    # Default values for the paths
    node_or_paths = [0, "OR", label, [[], [], [], []]]
    node_and_paths = [0, "AND", label, [[], [], [], []]]
    node_and_not_paths = [0, "NOT", label, [[], [], [], []]]
    node_individual_path = [0, "INDIVIDUAL", label, [[], [], [], []]]
    if label_quantities is None:
        # Label discarded at the previous step
        return [
            node_individual_path,
            node_or_paths,
            node_and_paths,
            node_and_not_paths,
        ], 0.0

    # Label estimation (no further path. Base case scenario where we stop the search here)
    max_individual_iou, min_individual_iou = (
        optimal_utils.estimate_min_max_iou_from_label_info(
            label_quantities=label_quantities,
            neuron_quantities=neuron_quantities,
            num_hits=num_hits,
            minimum_threshold=minimum_threshold,
        )
    )
    if len(label) == max_length or next_path == "INDIVIDUAL":
        # If the label is already at maximum length, we cannot estimate any further path
        individual_path = [max_individual_iou, "INDIVIDUAL", label, [[], [], [], []]]
        return (
            [individual_path, node_or_paths, node_and_paths, node_and_not_paths],
            min_individual_iou,
        )

    # We extract the "allowed" operators from this node.
    # This is the result of previous "paths" operations. See below
    if previous_paths_to_expand is not None:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = (
            previous_paths_to_expand
        )
    else:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = (
            [],
            [],
            [],
            [],
        )

    # Conditions to steer the estimation of the chains
    available_spots = max_length - len(label)
    or_condition = available_spots > 0 and (
        previous_paths_to_expand is None or is_in("OR", previous_or_paths)
    )
    and_condition = available_spots > 0 and (
        previous_paths_to_expand is None or is_in("AND", previous_and_paths)
    )
    and_not_condition = available_spots > 0 and (
        previous_paths_to_expand is None or is_in("NOT", previous_and_not_paths)
    )
    and_and_not_condition = available_spots > 1 and (
        previous_paths_to_expand is None
        or (
            is_in(["AND", "NOT"], previous_and_paths)
            or is_in(["AND", "NOT"], previous_and_not_paths)
        )
    )
    and_or_condition = available_spots > 1 and (
        previous_paths_to_expand is None
        or (
            is_in(["OR", "AND"], previous_or_paths)
            or is_in(["OR", "AND"], previous_and_paths)
        )
    )
    or_not_condition = available_spots > 1 and (
        previous_paths_to_expand is None
        or (
            is_in(["OR", "NOT"], previous_or_paths)
            or is_in(["OR", "NOT"], previous_and_not_paths)
        )
    )
    every_condition = available_spots > 2 and (
        previous_paths_to_expand is None
        or (
            is_in(["OR", "AND", "NOT"], previous_or_paths)
            or is_in(["OR", "AND", "NOT"], previous_and_paths)
            or is_in(["OR", "AND", "NOT"], previous_and_not_paths)
        )
    )

    ############### Exclusive paths ###############
    if or_condition:
        or_max_min_intersection, or_max_min_union = heuristic.or_chain_estimation(
            label=label,
            label_quantities=label_quantities,
            neuron_quantities=neuron_quantities,
            max_improvement=max_min_improvement,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
            max_length=max_length,
        )
        max_or_chain_iou, min_or_chain_iou = max_min_iou_from_union_intersection(
            or_max_min_intersection, or_max_min_union, minimum_threshold
        )
    else:
        max_or_chain_iou, min_or_chain_iou = 0.0, 0.0

    # This covers AND chains
    if and_condition:
        and_max_min_intersection, and_max_min_union = heuristic.and_chain_estimation(
            label_quantities=label_quantities,
            neuron_quantities=neuron_quantities,
            max_improvement=max_min_improvement,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
        )
        max_and_chain_iou, min_and_chain_iou = max_min_iou_from_union_intersection(
            and_max_min_intersection, and_max_min_union, minimum_threshold
        )
    else:
        max_and_chain_iou, min_and_chain_iou = 0.0, 0.0

    if and_not_condition:
        and_not_max_min_intersection, and_not_max_min_union = (
            heuristic.and_not_chain_estimation(
                label_quantities=label_quantities,
                neuron_quantities=neuron_quantities,
                max_improvement=max_min_improvement,
                num_hits=num_hits,
                max_size_mask=max_size_mask,
            )
        )
        max_and_not_chain_iou, min_and_not_chain_iou = (
            max_min_iou_from_union_intersection(
                and_not_max_min_intersection, and_not_max_min_union, minimum_threshold
            )
        )
    else:
        max_and_not_chain_iou, min_and_not_chain_iou = 0.0, 0.0

    ############### Combined paths ###############
    if and_and_not_condition:
        max_and_and_not_chain_iou = max_and_chain_iou
        min_and_and_not_chain_iou = min_and_chain_iou
    else:
        max_and_and_not_chain_iou, min_and_and_not_chain_iou = 0.0, 0.0

    if and_or_condition:
        # This covers both AND-OR chaing and everything chain
        max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = estimate_pairwise_path(
            and_max_min_intersection,
            and_max_min_union,
            or_max_min_intersection,
            or_max_min_union,
            minimum_threshold=minimum_threshold,
        )
    else:
        max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0

    if or_not_condition:
        max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = (
            estimate_pairwise_path(
                or_max_min_intersection,
                or_max_min_union,
                and_not_max_min_intersection,
                and_not_max_min_union,
                minimum_threshold=minimum_threshold,
            )
        )
    else:
        max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0

    # Greater chains
    if every_condition:
        max_every_chain_iou = max_comb_and_or_chain_iou
        min_every_chain_iou = min_comb_and_or_chain_iou
    else:
        max_every_chain_iou, min_every_chain_iou = 0.0, 0.0

    # Update the mininum threshold if there is a new minimum that is greater than the current threshold
    # The minimum is used only for this update
    max_minimum = max(
        min_individual_iou,
        min_or_chain_iou,
        min_and_chain_iou,
        min_and_not_chain_iou,
        min_comb_and_or_chain_iou,
        min_comb_or_andnot_chain_iou,
    )
    minimum_threshold = max(minimum_threshold, max_minimum)

    # Update the paths with the new estimations if they are above the minimum threshold
    # Note that for the combined paths we do not create a new "combined path" to avoid the combinatorial explosion.
    # Instead we update the estimation of the exclusive paths if the combined path achieves an estimation greater than the exclusive path.
    # The resulting node will include which "operators" are allowed: if it is an exclusive operator it will have just 1 operator. Otherwise multiple
    for ops, max_iou, _ in zip(
        [
            [],
            ["OR"],
            ["AND"],
            ["AND", "NOT"],
            ["NOT"],
            ["AND", "OR"],
            ["AND", "OR", "NOT"],
            ["OR", "NOT"],
        ],
        [
            max_individual_iou,
            max_or_chain_iou,
            max_and_chain_iou,
            max_and_and_not_chain_iou,
            max_and_not_chain_iou,
            max_comb_and_or_chain_iou,
            max_every_chain_iou,
            max_comb_or_andnot_chain_iou,
        ],
        [
            min_individual_iou,
            min_or_chain_iou,
            min_and_chain_iou,
            min_and_and_not_chain_iou,
            min_and_not_chain_iou,
            min_comb_and_or_chain_iou,
            min_every_chain_iou,
            min_comb_or_andnot_chain_iou,
        ],
    ):
        if max_iou > 0 and max_iou >= minimum_threshold:
            if len(ops) == 0:
                node_individual_path[INDEX_NODE_IOU_ESTI] = max_iou
            elif len(ops) == 1:
                op = ops[0]
                if op == "OR":
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                elif op == "AND":
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                elif op == "NOT":
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                elif op == "INDIVIDUAL":
                    node_individual_path[INDEX_NODE_IOU_ESTI] = max_iou
            elif len(ops) == 2:
                if "OR" in ops and "AND" in ops:
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                elif "OR" in ops and "NOT" in ops:
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                elif "AND" in ops and "NOT" in ops:
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
            elif len(ops) == 3:
                if "OR" in ops and "AND" in ops and "NOT" in ops:
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
            else:
                raise ValueError(f"Unexpected number of operations: {len(ops)}")

    # Return all the paths and the maximum among the minimum
    max_results = [
        node_individual_path,
        node_or_paths,
        node_and_paths,
        node_and_not_paths,
    ]
    return max_results, max_minimum
