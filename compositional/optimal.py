"""Optimal-search implementation for compositional neuron explanations.
"""

import heapq

import numpy as np

from compositional import formula as F
from compositional import path_heuristic
from utils import optimal_utils, mask_utils
from utils.constants import INDEX_OR, INDEX_AND, INDEX_NOT


def update_frontier_by_ancestors(
    frontier,
    ancestors,
    *,
    threshold,
    label_mapping,
    heuristic_info,
    max_min_improvement,
    disjoint_info,
    num_hits,
    max_size_mask,
    max_length,
):
    """Update the frontier by re-estimating the IoU of the formulas in the frontier based on the new information from the ancestors.

    Args:
        frontier (list): A list of formulas in the frontier. Each element is a tuple containing the estimated IoU, the next operation, the current label, the paths to expand for the next iterations, and the heuristic used for the estimation.
        ancestors (list): A list of ancestor labels that have new information to update the frontier with.
        threshold (float): The current minimum threshold for the IoU estimation. This is used to filter out formulas that have an estimated IoU lower than this threshold.
        label_mapping (dict): A dictionary mapping the labels to their indices in the masks_info and quantities.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        max_min_improvement (list): A list of tuples containing the maximum and minimum improvement for each quantity and each length of explanation. Each tuple contains the maximum and minimum improvement for the sample and the sum.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        num_hits (int): The number of hits of the bitmap.
        max_size_mask (int): The maximum size of the masks.
        max_length (int): The maximum length of the explanations.

    Returns:
        new_frontier (list): A list of formulas in the updated frontier. Each element is a tuple containing the estimated IoU, the next operation, the current label, the paths to expand for the next iterations, and the heuristic used for the estimation. The list is sorted by the estimated IoU in descending order.
        new_threshold (float): The updated minimum threshold for the IoU estimation. This is the maximum between the input threshold and the minimum IoU estimated for the formulas in the frontier.
    """
    sorted_ancestors = sorted(ancestors, key=lambda x: len(x))
    new_frontier = []
    found_ancestor = False
    for frontier_node in frontier:
        node_label = frontier_node[2]
        node_tree_path = node_label.tree_path()
        node_len = len(node_label)
        node_heuristic = frontier_node[4]
        found_ancestor = False
        for ancestor_label in sorted_ancestors:
            if found_ancestor:
                # If we already found an ancestor, we can stop checking the others since the other matches will be with shorter common ancestors
                break

            if (len(ancestor_label) == node_len and ancestor_label == node_label) or (
                len(ancestor_label) < node_len and ancestor_label in node_tree_path
            ):
                # We compute the estimation per sample since this is the only time we have access to this information
                new_max, new_min = path_heuristic.update_paths_iou(
                    heuristic_name="sample",
                    node=frontier_node,
                    label_mapping=label_mapping,
                    heuristic_info=heuristic_info,
                    max_improvement=max_min_improvement,
                    disjoint_info=disjoint_info,
                    num_hits=num_hits,
                    max_size_mask=max_size_mask,
                    max_length=max_length,
                    minimum_threshold=threshold,
                )

                if new_max >= threshold:
                    # Add it back no matter if the new max is unchanged/
                    # Note the new estimation cannot be greater than the previous one since it will have the same or better info
                    new_frontier.append(
                        (
                            -new_max,
                            frontier_node[1],
                            frontier_node[2],
                            frontier_node[3],
                            node_heuristic,
                        )
                    )

                # Update the minimum threshold
                if new_min > threshold:
                    threshold = new_min
                # We can exit because the other matches will produce the same results since the heuristics info do not change
                found_ancestor = True

        if not found_ancestor:
            # If no ancestor was found, we add the node to the past frontier
            new_frontier.append(frontier_node)
    heapq.heapify(new_frontier)
    return new_frontier, threshold


def apply_distributive_property(node):
    """Apply the Distributive Property to the label if possible.

    The Distributive Property states that A AND (B OR C) is equivalent to (A AND B) OR (A AND C) and A OR (B AND C) is equivalent to (A OR B) AND (A OR C). This transformation is used to correct the overestimation of the heuristics in some cases. For example, if you have a formula like (A OR B) AND NOT C, the heuristic might overestimate the IoU of this formula because it does not consider the fact that NOT C can be distributed to both A and B. By applying the Distributive Property, we can transform this formula into ((A AND NOT C) OR (B AND NOT C)), which has a more accurate estimation of the IoU.

    Args:
        node (tuple): A tuple containing the information of the node to apply the Distributive Property to. The tuple contains the estimated IoU, the next operation, the current label, the paths to expand for the next iterations, and the heuristic used for the estimation.

    Returns:
        new_label (F.Formula): The new label after applying the Distributive Property. If the Distributive Property cannot be applied, the original label is returned.
    """
    _, next_op_node, label, _, _ = node
    if next_op_node != "INDIVIDUAL" or len(label) < 3:
        # We apply this to only labels and not paths.
        # This is because the purpose of this transformation is to correct
        # errors in overstimations cumulated through the paths. In this case, often
        # You have something like (A OR B) AND NOT C with NOT C dominating for the max improvement factor
        # Since the distributive property ensures that the semantic is preserved, we try to correct those overestimations by
        # considering the transformations since computing ((A AND NOT C) OR (B AND NOT C)) is more accurate than
        # computing ((A OR B) AND NOT C) since everythin is a pariwise estimation.
        return label
    label_left = label.left
    num_left_op = len(set(label_left.get_ops()))
    if num_left_op == 1:
        external_op = label.__class__
        internal_op = label_left.__class__
        if external_op == internal_op:
            return label
        # We can apply the Distributive Property
        vals_left = label_left.get_vals()
        chain = []
        for val in vals_left:
            inner_formula = external_op(F.Leaf(val), label.right)
            chain.append(inner_formula)
        for i in range(len(chain) - 1):
            final_formula = internal_op(chain[i], chain[i + 1])
        return final_formula
    else:
        return label


def extract_max_min_quantity_improvement(
    quantity,
    *,
    concepts,
    concepts_quantities,
    label_mapping,
    length,
    sample_limits,
    sum_limits,
):
    """Extract the maximum and minimum improvement for a given quantity and a given length of explanation. The improvement is computed by summing the top-k and bottom-k values of the quantity across samples and concepts. The improvement is then limited by the sample limits and sum limits.

    Args:
        quantity (str): The name of the quantity to extract.
        concepts (list): A list of concepts to extract the quantity from.
        concepts_quantities (list): A list of dictionaries containing the quantities for each concept.
        label_mapping (dict): A dictionary mapping the labels to their indices in the masks_info and quantities.
        length (int): The length of the explanation to extract the improvement for.
        sample_limits (list): A list of limits for the improvement per sample. The improvement per sample is limited by the corresponding value in the sample limits.
        sum_limits (list): A list of limits for the improvement sum. The improvement sum is limited by the corresponding value in the sum limits.

    Returns:
        max_min_improvement (tuple): A tuple containing the maximum and minimum improvement for the given quantity and length of explanation. The first element of the tuple is the maximum improvement and the second element is the minimum improvement. Each improvement is a tuple containing the improvement per sample and the improvement sum.
    """
    list_quantity = []
    for label in concepts:
        label_quantities = concepts_quantities[label_mapping[label]]
        list_quantity.append(
            optimal_utils.get_quantity(
                label_info=label_quantities,
                quantity_name=quantity,
                quantity_type="max",
                quantity_scope="sample",
            )
        )

    quantity_vector = np.stack(list_quantity, axis=0)

    # Sort each quantity by the sum of the values across samples (per concept)
    sum_quantity = quantity_vector.sum(axis=1)
    sorted_quantity = -np.sort(-sum_quantity)

    # For each sample, extract the top-k and bottom-k values for each quantity (per sample)
    topk_quantity = np.partition(quantity_vector, -length, axis=0)[-length:]
    bottom_quantity = np.partition(quantity_vector, length - 1, axis=0)[:length]

    # For each length of explanation, compute the maximum and minimum improvement
    max_min_improvement_lens = []
    for explanation_len in range(1, length + 2):
        max_improvement_sample = topk_quantity[:explanation_len].sum(
            axis=0
        )  # Max improvement is the sum of the top-k values
        min_improvement_sample = bottom_quantity[:explanation_len].sum(
            axis=0
        )  # Min improvement is the sum of the bottom-k values
        max_improvement_sum = sum(sorted_quantity[:explanation_len])  # Max improvement
        min_improvement_sum = sum(sorted_quantity[-explanation_len:])  # Min improvement
        # Ensure the improvement respects the limits
        for sample_limit in sample_limits:
            max_improvement_sample = np.minimum(max_improvement_sample, sample_limit)
            min_improvement_sample = np.minimum(min_improvement_sample, sample_limit)
        for sum_limit in sum_limits:
            max_improvement_sum = min(max_improvement_sum, sum_limit)
            min_improvement_sum = min(min_improvement_sum, sum_limit)
        len_max_improvement = (
            # Per sample
            max_improvement_sample,
            min_improvement_sample,
            # Sums
            max_improvement_sum,
            min_improvement_sum,
        )
        max_min_improvement_lens.append(len_max_improvement)

    return max_min_improvement_lens


def compute_max_min_improvement(
    heuristic_info, label_mapping, max_size_mask, num_hits, length
):
    """Compute the maximum and minimum improvement for each quantity and each length of explanation. This is used to estimate the paths IoU of the formulas in the frontier.

    Args:
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        label_mapping (dict): A dictionary mapping the labels to their indices in the masks_info and quantities.
        max_size_mask (int): The maximum size of the masks.
        num_hits (int): The number of hits of the bitmap.
        length (int): The maximum length of the explanations.

    Returns:
        max_min_improvement (list): A list of tuples containing the maximum and minimum improvement for each quantity and each length of explanation. Each tuple contains the maximum and minimum improvement for the sample and the sum.
    """

    # Decompose quantities
    _, neuron_quantities, concepts_quantities = heuristic_info
    (
        neuron_unique_tuple,
        neuron_common_tuple,
        neuron_coverable_tuple,
        neuron_sum_tuple,
        _,
        _,
    ) = neuron_quantities
    neuron_unique = neuron_unique_tuple[0]
    neuron_common = neuron_common_tuple[0]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_coverable_sum = neuron_coverable_tuple[1]
    neuron_common_sum = neuron_common_tuple[1]
    neuron_unique_sum = neuron_unique_tuple[1]
    neuron_sum = neuron_sum_tuple[0]

    list_concepts = list(label_mapping.keys())

    # Limit for sums
    tot_size = max_size_mask * neuron_common.shape[0]

    # Extract the quantities for each concept
    improvement_info_common_intersection = extract_max_min_quantity_improvement(
        "common_intersection",
        concepts=list_concepts,
        concepts_quantities=concepts_quantities,
        label_mapping=label_mapping,
        length=length,
        sample_limits=[neuron_common, neuron_coverable],
        sum_limits=[neuron_common_sum, neuron_coverable_sum],
    )
    improvement_info_unique_intersection = extract_max_min_quantity_improvement(
        "unique_intersection",
        concepts=list_concepts,
        concepts_quantities=concepts_quantities,
        label_mapping=label_mapping,
        length=length,
        sample_limits=[neuron_unique, neuron_coverable],
        sum_limits=[neuron_unique_sum, neuron_coverable_sum],
    )
    improvement_info_common_extras = extract_max_min_quantity_improvement(
        "common_extras",
        concepts=list_concepts,
        concepts_quantities=concepts_quantities,
        label_mapping=label_mapping,
        length=length,
        sample_limits=[max_size_mask - neuron_sum],
        sum_limits=[tot_size - num_hits],
    )
    improvement_info_unique_extras = extract_max_min_quantity_improvement(
        "unique_extras",
        concepts=list_concepts,
        concepts_quantities=concepts_quantities,
        label_mapping=label_mapping,
        length=length,
        sample_limits=[max_size_mask - neuron_sum],
        sum_limits=[tot_size - num_hits],
    )
    improvement_info_common_uncovered = extract_max_min_quantity_improvement(
        "common_uncovered",
        concepts=list_concepts,
        concepts_quantities=concepts_quantities,
        label_mapping=label_mapping,
        length=length,
        sample_limits=[neuron_coverable],
        sum_limits=[neuron_coverable_sum],
    )
    improvement_info_unique_uncovered = extract_max_min_quantity_improvement(
        "unique_uncovered",
        concepts=list_concepts,
        concepts_quantities=concepts_quantities,
        label_mapping=label_mapping,
        length=length,
        sample_limits=[neuron_coverable],
        sum_limits=[neuron_coverable_sum],
    )
    max_min_improvement_info = [
        improvement_info_common_intersection,
        improvement_info_unique_intersection,
        improvement_info_common_extras,
        improvement_info_unique_extras,
        improvement_info_common_uncovered,
        improvement_info_unique_uncovered,
    ]

    return max_min_improvement_info


def estimate_iou_frontier(
    *,
    frontier,
    label_mapping,
    heuristic,
    heuristic_info,
    max_min_improvement,
    disjoint_info,
    num_hits,
    max_size_mask,
    length,
    global_min_threshold,
):
    """Estimate the IoU of the formulas in the frontier using the heuristic. The estimation is used to sort the frontier and to filter out formulas that have an estimated IoU lower than the global minimum threshold.

    Args:
        frontier (list): A list of formulas to estimate.
        label_mapping (dict): A dictionary mapping the labels to their indices in the masks_info and quantities.
        heuristic (str): The name of the heuristic to use for the estimation.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        max_min_improvement (list): A list of tuples containing the maximum and minimum improvement for each quantity and each length of explanation. Each tuple contains the maximum and minimum improvement for the sample and the sum.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        num_hits (int): The number of hits of the bitmap.
        max_size_mask (int): The maximum size of the masks.
        length (int): The maximum length of the explanations.
        global_min_threshold (float): The global minimum threshold for the IoU estimation. Formulas with an estimated IoU lower than this threshold will be filtered out.

    Returns:
        frontier_estimates (list): A list of tuples containing the estimated IoU and the corresponding formula for each formula in the frontier. The list is sorted by the estimated IoU in descending order.
        minimum_threshold (float): The minimum threshold for the IoU estimation. This is the maximum between the global minimum threshold and the minimum IoU estimated for the formulas in the frontier.
    """

    # Sort the frontier based on the heuristic score and store both the node of the frontier and the score
    frontier_estimates = []
    minimum_threshold = global_min_threshold

    for node in frontier:
        # Estimate the heuristic score for the node
        max_score, min_score = path_heuristic.estimate_paths_iou(
            heuristic,
            node,
            label_mapping,
            heuristic_info,
            max_min_improvement=max_min_improvement,
            disjoint_info=disjoint_info,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
            max_length=length,
            minimum_threshold=minimum_threshold,
        )

        # If the minimum score is greater than the current minimum threshold, update the minimum threshold
        if min_score > minimum_threshold:
            minimum_threshold = min_score

        # Add nodes and their estimation to the current frontier
        label = node[2]
        for node_path in max_score:
            node_path_max_iou, node_path_next_op, _, node_path_paths_to_expand = (
                node_path
            )
            if node_path_max_iou > 0:
                # Add the path to the frontier estimates if the IoU is greater than 0.
                # Note: filtering by minimum_threshold is done inside the estimate_label_iou function
                frontier_estimates.append(
                    (
                        -node_path_max_iou,
                        node_path_next_op,
                        label,
                        node_path_paths_to_expand,
                        heuristic,
                    )
                )
                if node_path_max_iou < minimum_threshold:
                    raise ValueError(
                        f"Node {node} has a max IoU {node_path_max_iou} lower than the minimum threshold {minimum_threshold}. This should not happen."
                    )
    if minimum_threshold > global_min_threshold:
        # Reduce the frontier with the new minimum threshold
        # This case covers the time where the minimum threshold is found later in the search
        frontier_estimates = reduce_frontier(frontier_estimates, minimum_threshold)
    return frontier_estimates, minimum_threshold


def reduce_frontier(frontier, threshold):
    """Remove nodes from the frontier that have an estimated IoU lower than the global minimum threshold.

    Args:
        frontier (list): A list of candidate labels.
        global_minimum_threshold (float): The global minimum threshold.

    Returns:
        reduced_frontier (list): A reduced list of candidate labels.
    """
    reduced_frontier = []
    for node in frontier:
        iou = node[0]
        if -iou >= threshold:
            reduced_frontier.append(node)
    heapq.heapify(reduced_frontier)
    return reduced_frontier


def update_frontier(
    *,
    past_frontier,
    new_nodes,
    label_mapping,
    heuristic,
    heuristic_info,
    max_min_improvement,
    disjoint_info,
    num_hits,
    max_size_mask,
    length,
    global_min_threshold,
):
    """Update the frontier with the new nodes and sort it based on the estimated IoU. The estimation is used to filter out formulas that have an estimated IoU lower than the global minimum threshold.

    Args:
        past_frontier (list): A list of formulas in the past frontier. This is used to merge the new nodes with the past frontier after the estimation. If None, the new frontier is initialized with the new nodes.
        new_nodes (list): A list of formulas to estimate and add to the frontier.
        label_mapping (dict): A dictionary mapping the labels to their indices in the masks_info and quantities.
        heuristic (str): The name of the heuristic to use for the estimation.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        max_min_improvement (list): A list of tuples containing the maximum and minimum improvement for each quantity and each length of explanation. Each tuple contains the maximum and minimum improvement for the sample and the sum.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        num_hits (int): The number of hits of the bitmap.
        max_size_mask (int): The maximum size of the masks.
        length (int): The maximum length of the explanations.
        global_min_threshold (float): The global minimum threshold for the IoU estimation. Formulas with an estimated IoU lower than this threshold will be filtered out.

    Returns:
        sorted_frontier (list): A list of tuples containing the estimated IoU and the corresponding formula for each formula in the frontier. The list is sorted by the estimated IoU in descending order.
        minimum_threshold (float): The minimum threshold for the IoU estimation. This is the maximum between the global minimum threshold and the minimum IoU estimated for the formulas in the frontier.

    """
    # Estimate the IoU of the new frontier
    new_frontier, local_minimum_threshold = estimate_iou_frontier(
        frontier=new_nodes,
        label_mapping=label_mapping,
        heuristic=heuristic,
        heuristic_info=heuristic_info,
        max_min_improvement=max_min_improvement,
        num_hits=num_hits,
        max_size_mask=max_size_mask,
        length=length,
        global_min_threshold=global_min_threshold,
        disjoint_info=disjoint_info,
    )

    # The new frontier computed an higher minimum threshold
    if local_minimum_threshold > global_min_threshold:
        # Update the global minimum threshold
        global_min_threshold = local_minimum_threshold
        # Reduce the past frontier based on the new global minimum threshold
        if past_frontier is not None:
            past_frontier = reduce_frontier(past_frontier, global_min_threshold)
    # Merge the new nodes with the past frontier
    if past_frontier is not None and len(past_frontier) > 0:
        # Merge assumes that the input iterables are sorted
        for new_node in new_frontier:
            heapq.heappush(past_frontier, new_node)
        sorted_frontier = past_frontier
    else:
        # Initialize frontier as a queue
        heapq.heapify(new_frontier)
        sorted_frontier = new_frontier
    return sorted_frontier, global_min_threshold


def is_path_included(index_op, list_to_check, list_to_add):
    """Check if the path to add is already included in the list of paths to check for a given operation.
    This is used to avoid adding duplicate paths to the frontier.
    Args:
        index_op (int): The index of the operation to check. This is used to select the list of paths to check for the given operation.
        list_to_check (list): A list of lists of paths to check for each operation.
        list_to_add (list): A list of operations representing the path to add to the frontier. This is used to check if the path to add is already included in the list of paths to check for the given operation.
    Returns:
        bool: True if the path to add is already included in the list of paths to check for the given operation, False otherwise.
    """
    for items in list_to_check[index_op]:
        if len(items) == len(list_to_add) and set(items) == set(list_to_add):
            # If the items are the same, we do not add them
            return True
    return False


def expand_node(frontier_node, *, candidate_labels, max_length):
    """Expand the node by adding a new candidate label to the formula based on the next operation. The expansion is done by creating a new formula with the candidate label and the current label based on the next operation. The new formulas are added to the frontier with the corresponding paths to expand for the next iterations. The paths to expand are generated based on the current paths to expand and the operations already present in the formula.

    Args:
        frontier_node (tuple): A tuple containing the information of the node to expand. The tuple contains the estimated IoU, the next operation, the current label, the paths to expand for the next iterations, and the heuristic used for the estimation.
        candidate_labels (list): A list of candidate labels to add to the formula.
        max_length (int): The maximum length of the formulas to search.

    Returns:
        next_frontier (list): A list of nodes containing the information relative to the labels generated by expanding the current node and the feasible paths. The other values in the node are set to None
    """
    _, next_op, label, paths_to_expand, _ = frontier_node
    next_frontier = []
    # Info useful to avoid logical equivalence
    if len(label) > 1:
        last_val = label.get_vals()[0]
        last_op = label.get_ops()[0]
    for candidate_term in candidate_labels:
        # Skip the candidate term if it is already in the label
        if candidate_term.val in label.get_vals():
            continue
        # Impose order to avoid logical equivalence
        if (
            next_op != "NOT"
            and isinstance(label, F.Leaf)
            and candidate_term.val < label.val
        ):
            # A OR B is equivalent to B or A, so we impose an order on the candidate terms to avoid generating both paths
            continue
        elif len(label) > 1:
            # If the label is not a leaf we avoid logical equivalence between the last two operators.
            # Looking beyond the last two would be costly
            if next_op == "NOT" and last_op == "AND" and isinstance(label.right, F.Not):
                if candidate_term.val < last_val:
                    continue
            elif last_op == next_op:
                if candidate_term.val < last_val:
                    continue

        # Build the candidate formula based on the next operation
        if next_op == "OR":
            candidate_formula = F.Or(label, candidate_term)
        elif next_op == "AND":
            candidate_formula = F.And(label, candidate_term)
        elif next_op == "NOT":
            candidate_formula = F.And(label, F.Not(candidate_term))
        else:
            raise ValueError(f"Unknown operation {next_op}")

        # Generate the possible paths
        if len(candidate_formula) == max_length:
            # If we reached the maximum lenght, we don't need any other path
            next_frontier.append((None, "INDIVIDUAL", candidate_formula, [], None))
        else:
            # Otherwise we need to generate the possible paths to expand for the next iterations.
            # We generate them based on the paths to expand (allowed ones) of the current node and we filter
            # them based on the operations already present in the formula and the available spots in
            # the formula before reaching the maximum length.
            current_ops = set(candidate_formula.get_ops())
            available_spots = max_length - len(candidate_formula)
            feasibles_paths = [[], [], [], []]

            # Given the current allowed paths, mantain only the feasible ones (i.e., the ones that can be fully explored given the steps left)
            for path in paths_to_expand:
                for path_ops in path:
                    path_missing_ops = set(path_ops) - current_ops

                    if len(path_missing_ops) == available_spots:
                        # In this case we have the same number of missing operations and available spaces
                        # The next operator to add needs to be one of the missing operations
                        path_to_add = list(path_missing_ops)
                    else:
                        # In this case we have more spaces available then missing operations
                        # We can freely choose the next operator
                        path_to_add = path_ops

                    for op_to_add in path_to_add:
                        if op_to_add == "OR":
                            index_op = INDEX_OR
                        elif op_to_add == "AND":
                            index_op = INDEX_AND
                        elif op_to_add == "NOT":
                            index_op = INDEX_NOT
                        else:
                            raise ValueError(f"Unknown operation {path_to_add}")

                        # Add this path if it is not already present in the feasible ops

                        if not is_path_included(index_op, feasibles_paths, path_to_add):
                            feasibles_paths[index_op].append(list(path_to_add))
            next_frontier.append((None, None, candidate_formula, feasibles_paths, None))
    return next_frontier


def perform_search(
    *, masks, bitmaps, heuristic_info, disjoint_info, max_size_mask, length=3
):
    """Perform the optimal search over compositional formulas.

    Args:
        masks (list): A list of concept masks indexed by concept id.
        bitmaps (torch.Tensor): A tensor of shape (N, H*W), where N is the number of samples.
        heuristic_info (tuple): A tuple containing masks info, neuron quantities, and concept quantities.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        max_size_mask (int): The maximum size of the mask.
        length (int): The maximum length of the formulas to search.

    Returns:
        tuple: `(best_label, best_iou, visited_nodes, expanded_nodes, estimated_nodes)`
            where `best_label` is a formula object or `None` if no valid formula is found,
            `best_iou` is a float, and the remaining values are integer search statistics.
    """

    # Number of hits
    num_hits = np.int64(bitmaps.sum().item())

    # Candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    label_mapping = {F.Leaf(c): c for c in range(len(masks))}

    # Initialize the frontier with all the candidate labels
    initial_frontier = [(0.0, None, k, None, None) for k in label_mapping.keys()]

    # Max improvement for this neuron
    max_min_improvement = compute_max_min_improvement(
        heuristic_info, label_mapping, max_size_mask, num_hits, length
    )

    # Statistics
    expanded_nodes = 0
    estimated_nodes = 0
    visited_nodes = 0

    # Aux
    best_results = (0.0, None)  # (IoU, label)
    recent_nodes = []
    visited = []
    recent_e_iou = 2
    minimum_threshold = 0.0

    # Initialize frontiers
    current_frontier, minimum_threshold = update_frontier(
        past_frontier=None,
        new_nodes=initial_frontier,
        label_mapping=label_mapping,
        heuristic="sum",
        heuristic_info=heuristic_info,
        max_min_improvement=max_min_improvement,
        num_hits=num_hits,
        max_size_mask=max_size_mask,
        length=length,
        global_min_threshold=minimum_threshold,
        disjoint_info=disjoint_info,
    )

    done = len(current_frontier) == 0
    while not done:
        # Get the node with the highest estimated IoU from the frontier
        node = heapq.heappop(current_frontier)
        e_node, next_op_node, label_node, _, node_heuristic = node
        if -e_node < minimum_threshold:
            # Unuseful node, skip it
            done = len(current_frontier) == 0
            continue

        print(
            f"Heuristic: {-e_node:<.3f}, Minimum Threshold: {minimum_threshold:<.3f}, Frontier Size: {len(current_frontier)} Best Node: {best_results}",
            end="\r",
        )

        # Compute Sample Estimate
        if node_heuristic != "sample":
            new_max, _ = path_heuristic.update_paths_iou(
                heuristic_name="sample",
                node=node,
                label_mapping=label_mapping,
                heuristic_info=heuristic_info,
                max_improvement=max_min_improvement,
                disjoint_info=disjoint_info,
                num_hits=num_hits,
                max_size_mask=max_size_mask,
                max_length=length,
                minimum_threshold=minimum_threshold,
            )
            if new_max < -e_node:
                # If the estimate is lower than the previous one
                if new_max > minimum_threshold:
                    # we update it and reinsert the node in the frontier
                    heapq.heappush(
                        current_frontier,
                        (-new_max, next_op_node, label_node, node[3], "sample"),
                    )
                done = len(current_frontier) == 0
                continue

        # For label of size >=3 we can try to apply the Distributive Property
        transformed_label = apply_distributive_property(node)
        if transformed_label != label_node:
            # If we can apply the distributive property,
            # we re-estimate its IoU
            node_after_distr = (node[0], node[1], transformed_label, node[3])
            new_max, new_min = path_heuristic.update_paths_iou(
                heuristic_name="sample",
                node=node_after_distr,
                label_mapping=label_mapping,
                heuristic_info=heuristic_info,
                max_improvement=max_min_improvement,
                disjoint_info=disjoint_info,
                num_hits=num_hits,
                max_size_mask=max_size_mask,
                max_length=length,
                minimum_threshold=minimum_threshold,
            )

            # If the new estimate is lower than the previous one we update it
            # and reinsert the node in the frontier. Otherwise, we keep analyzing the previous node
            if new_max < -e_node:
                if new_max > minimum_threshold:
                    # If the estimate is lower than the previous one we update it and reinsert the node in the frontier
                    heapq.heappush(
                        current_frontier,
                        (-new_max, next_op_node, label_node, node[3], "sample"),
                    )

                if new_min > minimum_threshold:
                    # If the new minimum is greater than the current minimum threshold, we update the minimum threshold
                    minimum_threshold = new_min
                    current_frontier = reduce_frontier(
                        current_frontier, minimum_threshold
                    )
                done = len(current_frontier) == 0
                continue

        # Recent node mechanism to avoid expanding the same node multiple times
        if -e_node >= recent_e_iou:
            if node in recent_nodes:
                done = len(current_frontier) == 0
                continue
            else:
                recent_nodes.append(node)
        else:
            recent_nodes = [node]
            recent_e_iou = -e_node

        # "Final" node case, without any path to expand
        if next_op_node == "INDIVIDUAL":
            if label_node not in visited:

                # Get the formula mask and its tree
                tree_formula_masks = mask_utils.get_formula_mask_and_tree(
                    label_node, masks, device=bitmaps.device
                )
                label_mask = tree_formula_masks[label_node]

                # Compute the info and iou for the current node
                label_info, label_iou = (
                    optimal_utils.compute_label_info_and_iou_from_mask(
                        label_mask,
                        bitmaps=bitmaps,
                        heuristic_info=heuristic_info,
                        num_hits=num_hits,
                    )
                )

                # Add the node to the visited nodes
                visited_nodes += 1
                visited.append(label_node)

                # Update the results and the minimum threshold if needed
                if label_iou > best_results[0]:
                    best_results = (label_iou, label_node)
                elif label_iou == best_results[0]:
                    if len(label_node) < len(best_results[1]):
                        # We prefer shorter labels in case of equal IoU
                        best_results = (label_iou, label_node)
                if label_iou > minimum_threshold:
                    minimum_threshold = label_iou
                    current_frontier = reduce_frontier(
                        current_frontier, minimum_threshold
                    )

                #################### PROPAGATION ################################
                # Compute the info for ancestors
                ancestors_info = {}
                for ancestor, ancestor_mask in tree_formula_masks.items():
                    if ancestor == label_node or ancestor in label_mapping.keys():
                        # We already have these quantities
                        continue
                    else:
                        ancestor_mask = ancestor_mask.to(bitmaps.device)
                        ancestor_info, ancestor_iou = (
                            optimal_utils.compute_label_info_and_iou_from_mask(
                                ancestor_mask,
                                bitmaps=bitmaps,
                                heuristic_info=heuristic_info,
                                num_hits=num_hits,
                            )
                        )
                        # Add ancestor to visited nodes
                        visited.append(ancestor)
                        visited_nodes += 1

                        # Update the results and the minimum threshold if needed
                        if ancestor_iou > best_results[0]:
                            best_results = (ancestor_iou, ancestor)
                        elif ancestor_iou == best_results[0]:
                            if len(ancestor) < len(best_results[1]):
                                # We prefer shorter labels in case of equal IoU
                                best_results = (ancestor_iou, ancestor)
                        if ancestor_iou > minimum_threshold:
                            minimum_threshold = ancestor_iou
                            current_frontier = reduce_frontier(
                                current_frontier, minimum_threshold
                            )
                        ancestors_info[ancestor] = ancestor_info

                # Propagate the information to other nodes
                if len(label_node) < length:
                    # If the label is at maximum length, there is no need to record its info and backpropagate them
                    ancestors_info[label_node] = label_info

                # We compute temporary heuristic info and label mapping to update the frontier with the new info from ancestors.
                # We do not store them in the standard heuristic info and label mapping to avoid memory issues on extreme cases where
                # there is large exploration of the state space.
                # To speed up the process you can consider to replace the temp with the standard ones
                temp_heuristic_info, temp_label_mapping = (
                    optimal_utils.update_heuristic_info(
                        nodes_info=ancestors_info,
                        heuristic_info=heuristic_info,
                        label_mapping=label_mapping,
                        max_length=length,
                    )
                )

                # Update the frontier with the new info from ancestors
                current_frontier, new_minimum_threshold = update_frontier_by_ancestors(
                    current_frontier,
                    ancestors_info.keys(),
                    threshold=minimum_threshold,
                    label_mapping=temp_label_mapping,
                    heuristic_info=temp_heuristic_info,
                    max_min_improvement=max_min_improvement,
                    disjoint_info=disjoint_info,
                    num_hits=num_hits,
                    max_size_mask=max_size_mask,
                    max_length=length,
                )
                if new_minimum_threshold > minimum_threshold:
                    minimum_threshold = new_minimum_threshold
                    current_frontier = reduce_frontier(
                        current_frontier, minimum_threshold
                    )

            done = len(current_frontier) == 0
            continue

        # Expand the node to get the next frontier
        next_frontier = expand_node(
            node, candidate_labels=candidate_labels, max_length=length
        )
        expanded_nodes += 1
        estimated_nodes += len(next_frontier)

        # Compute the estimation for the next frontier
        current_frontier, minimum_threshold = update_frontier(
            past_frontier=current_frontier,
            new_nodes=next_frontier,
            label_mapping=label_mapping,
            heuristic="sum",
            heuristic_info=heuristic_info,
            max_min_improvement=max_min_improvement,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
            length=length,
            global_min_threshold=minimum_threshold,
            disjoint_info=disjoint_info,
        )

        done = len(current_frontier) == 0

    # Extract the best formula and its IoU
    best_iou = best_results[0]
    best_label = best_results[1]
    return best_label, best_iou, visited_nodes, expanded_nodes, estimated_nodes


def compute_optimal_explanations(*, bitmaps, masks, masks_info, disjoint_info, config):
    """Compute optimal explanations for the given neuron bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H*W), where N is the number of samples.
        masks (list): A list of concept masks indexed by concept id.
        masks_info (list): A list of dictionaries containing the information about the masks.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        config (Settings): A configuration object containing the search parameters.

    Returns:
        tuple: `(best_label, best_iou, visited_nodes, expanded_nodes, estimated_nodes)`.
    """
    # Extract info from config
    length = config.get_length()
    max_size_mask = config.get_mask_shape()[0] * config.get_mask_shape()[1]

    # Get parameters from config
    neuron_quantities, concept_quantities = optimal_utils.get_optimal_heuristic_info(
        masks=masks, bitmaps=bitmaps, masks_quantities=masks_info
    )

    heuristic_info = (
        masks_info,
        neuron_quantities,
        concept_quantities,
    )
    best_label, best_iou, visited, expanded, estimated = perform_search(
        masks=masks,
        bitmaps=bitmaps,
        heuristic_info=heuristic_info,
        disjoint_info=disjoint_info,
        max_size_mask=max_size_mask,
        length=length,
    )
    return best_label, best_iou, visited, expanded, estimated
