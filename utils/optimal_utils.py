"""Utilities for optimal-heuristic quantities and IoU bound computation.

Provides quantity extraction, label-info updates, and heuristic state updates.
"""

import torch
import numpy as np

from compositional import formula as F
from utils import general_utils
from utils.constants import (
    QUANTITIES,
    INDEX_TUPLE_MAX,
    INDEX_TUPLE_MIN,
    INDEX_TUPLE_SAMPLE,
    INDEX_TUPLE_SUM,
)


def get_quantity(*, label_info, quantity_name, quantity_type, quantity_scope):
    """Gets the quantity from the label info.

    Args:
        label_info (tuple): A tuple containing the common intersection, unique intersection, common extras, unique extras, and uncovered quantities for the label.
        quantity_name (str): The name of the quantity to get. Must be one of 'common_intersection', 'unique_intersection', 'common_extras', 'unique_extras', 'common_uncovered', 'unique_uncovered'.
        quantity_type (str): The type of the quantity to get. Must be one of 'max' or 'min'.
        quantity_scope (str): The scope of the quantity to get. Must be one of 'sample' or 'sum'.

    Returns:
        torch.Tensor: The requested quantity.
    """
    if quantity_scope not in ["sample", "sum"]:
        raise ValueError(f"Unknown quantity scope: {quantity_scope}")
    if quantity_name not in QUANTITIES:
        raise ValueError(f"Unknown quantity name: {quantity_name}")

    index_quantity = QUANTITIES.index(quantity_name)
    index_scope = INDEX_TUPLE_SAMPLE if quantity_scope == "sample" else INDEX_TUPLE_SUM

    if quantity_name == "unique_extras" or quantity_name == "unique_intersection":
        # For unique quantities, there is no max and min since they can be computed exactly
        quantity = label_info[index_quantity][index_scope]
    else:
        if quantity_type not in ["max", "min"]:
            raise ValueError(f"Unknown quantity type: {quantity_type}")
        index_type = INDEX_TUPLE_MAX if quantity_type == "max" else INDEX_TUPLE_MIN
        quantity = label_info[index_quantity][index_type][index_scope]

    return quantity


def get_concept_info(concept_quantities):
    """Gets the concept info from the concept quantities.
    Args:
        concept_quantities (tuple): A tuple containing the common intersection, unique intersection, common extras, unique extras, and uncovered quantities.
    Returns:
        tuple: A tuple containing the common intersection, unique intersection, common extras, unique extras, and uncovered quantities.
    """
    (
        common_intersection,
        unique_intersection,
        common_extras,
        unique_extras,
        common_uncovered,
        unique_uncovered,
    ) = concept_quantities
    common_intersection_sum = common_intersection.sum()
    unique_intersection_sum = unique_intersection.sum()
    common_extras_sum = common_extras.sum()
    unique_extras_sum = unique_extras.sum()
    common_uncovered_sum = common_uncovered.sum()
    unique_uncovered_sum = unique_uncovered.sum()

    # Min and Max are equal since they concept quantities are assumed to be exact
    tuple_common_intersection = (
        (common_intersection, common_intersection_sum),
        (common_intersection, common_intersection_sum),
    )
    tuple_unique_intersection = (unique_intersection, unique_intersection_sum)


    tuple_common_extras = (
        (common_extras, common_extras_sum),
        (common_extras, common_extras_sum),
    )

    tuple_unique_extras = (unique_extras, unique_extras_sum)

    
    tuple_common_uncovered = (
        (common_uncovered, common_uncovered_sum),
        (common_uncovered, common_uncovered_sum),
    )
    tuple_unique_uncovered = (
        (unique_uncovered, unique_uncovered_sum),
        (unique_uncovered, unique_uncovered_sum),
    )
    info = (
        tuple_common_intersection,
        tuple_unique_intersection,
        tuple_common_extras,
        tuple_unique_extras,
        tuple_common_uncovered,
        tuple_unique_uncovered,
    )
    return info


def compute_quantities_vector(
    *,
    label_mask,
    bitmaps,
    common_elements,
    unique_elements,
    neuron_common,
    neuron_unique,
):
    """Computes the quantities of the concept mask with respect to the bitmaps.

    Args:
        label_mask (torch.Tensor): The mask of the concept.
        bitmaps (torch.Tensor): The bitmaps of the dataset
        common_elements (torch.Tensor): The common elements in the bitmaps.
        unique_elements (torch.Tensor): The unique elements in the bitmaps.
        neuron_common (torch.Tensor): The common elements for the neuron.
        neuron_unique (torch.Tensor): The unique elements for the neuron.
    Returns:
        tuple: A tuple containing the common intersection, unique intersection,
                common extras, unique extras, and uncovered quantities.

    """
    num_elem = torch.numel(torch.ones_like(bitmaps))
    # Choose the dtype based on the number of elements
    if num_elem < 2**16:
        dtype = torch.int16
    elif num_elem < 2**32:
        dtype = torch.int32
    else:
        dtype = torch.int64
    if label_mask.device != bitmaps.device:
        label_mask = label_mask.to(bitmaps.device)
    intersection = label_mask & bitmaps
    unique_intersection = (
        (intersection & unique_elements).sum(dim=1, dtype=dtype).to("cpu").numpy()
    )
    common_intersection = (
        (intersection & common_elements).sum(dim=1, dtype=dtype).to("cpu").numpy()
    )
    extras = label_mask & (~bitmaps)
    common_extras = (extras & common_elements).sum(dim=1, dtype=dtype).to("cpu").numpy()
    unique_extras = (extras & unique_elements).sum(dim=1, dtype=dtype).to("cpu").numpy()
    common_uncovered = neuron_common - common_intersection
    unique_uncovered = neuron_unique - unique_intersection

    return (
        common_intersection,
        unique_intersection,
        common_extras,
        unique_extras,
        common_uncovered,
        unique_uncovered,
    )


def formula_all_disjoint(label, disjoint_info):
    """Check if the formula contains all disjoint labels."""
    vals = label.get_vals()
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if not are_disjoint(F.Leaf(vals[i]), F.Leaf(vals[j]), disjoint_info):
                return False
    return True


def are_disjoint(formula_a, formula_b, disjoint_info):
    """Checks if two formulas are disjoint based on the disjoint info.
    Args:
        formula_a (F.Formula): The first formula.
        formula_b (F.Formula): The second formula.
        disjoint_info (list): A matrix containing the disjoint information.
    Returns:
        bool: True if the formulas are disjoint, False otherwise.
    """
    if disjoint_info is None:
        return False
    if isinstance(formula_b, F.Not):
        return are_disjoint(formula_a, formula_b.val, disjoint_info)
    # TEMPORARY FOR DISTRIBUTIVE PROPERTY
    if not isinstance(formula_b, F.Leaf):
        return False
    concept_b = formula_b.val
    if isinstance(formula_a, F.Leaf):
        concept_a = formula_a.val
        if disjoint_info[concept_a, concept_b]:
            return True
        else:
            return False
    elif isinstance(formula_a, F.Or):
        left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
        right_disjoint = are_disjoint(formula_a.right, formula_b, disjoint_info)
        return left_disjoint and right_disjoint
    elif isinstance(formula_a, F.And):
        if isinstance(formula_a.right, F.Not):
            # In this case, we check if the left part is disjoint with the right part
            left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
            return left_disjoint
        else:
            # In this case, we check if the left or right part is disjoint. One is enough becase they limit each other
            left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
            right_disjoint = are_disjoint(formula_a.right, formula_b, disjoint_info)
        return left_disjoint or right_disjoint
    else:
        raise ValueError(f"Unknown formula type: {type(formula_a)}")


def estimate_label_quantities(
    heuristic, *, label, label_mapping, heuristic_info, max_size_mask, disjoint_info
):
    """Estimates the quantities for a given label based on the heuristic information.
    Args:
        heuristic (OptimalHeuristic): The heuristic to use for the estimation.
        label (F.Formula): The label to estimate the quantities for.
        label_mapping (dict): A dictionary mapping formulas to their index in the heuristic info.
        heuristic_info (tuple): A tuple containing the heuristic information for the concepts and neurons.
        max_size_mask (int): The maximum size of the mask that can be handled by the heuristic. If the label is larger than this size, it will not be estimated and None will be returned.
        disjoint_info (list): A matrix containing the disjoint information for the concepts. It is used to determine if two formulas are disjoint, which can simplify the estimation of their quantities.
    Returns:
        dict: A dictionary containing the estimated quantities for the label.
    """
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    if len(label) == 1 or label in label_mapping:
        # We already have this label in the heuristic info. No need to estimate it
        index_node = label_mapping[label]
        return concepts_quantities[index_node]

    # Extract heuristic information for the labels included in the node
    left_sublabel = label.left
    right_sublabel = label.right

    # For the not operator we extract its positive value since the negation is considered in the heuristic
    if isinstance(left_sublabel, F.Not):
        left_sublabel = left_sublabel.val
    if isinstance(right_sublabel, F.Not):
        right_sublabel = right_sublabel.val

    # Compute quantities for the left and right sublabels
    left_quantities = estimate_label_quantities(
        heuristic,
        label=left_sublabel,
        label_mapping=label_mapping,
        heuristic_info=heuristic_info,
        max_size_mask=max_size_mask,
        disjoint_info=disjoint_info,
    )
    if left_quantities is None:
        return None
    right_quantities = estimate_label_quantities(
        heuristic,
        label=right_sublabel,
        label_mapping=label_mapping,
        heuristic_info=heuristic_info,
        max_size_mask=max_size_mask,
        disjoint_info=disjoint_info,
    )
    if right_quantities is None:
        return None

    # Estimate whole label quantities from the sublabels quantities
    disjoint = are_disjoint(
        formula_a=label.left, formula_b=label.right, disjoint_info=disjoint_info
    )
    if disjoint:
        node_quantities = heuristic.estimate_disjoint_label_info(
            label, left_quantities=left_quantities, right_quantities=right_quantities
        )
    else:
        node_quantities = heuristic.estimate_label_info(
            label,
            left_quantities=left_quantities,
            right_quantities=right_quantities,
            neuron_quantities=neuron_quantities,
        )

    return node_quantities


def get_neuron_quantities(
    bitmaps, common_elements, unique_elements, uncoverable_elements
):
    """Gets the neuron quantities for the optimal heuristic.
    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        common_elements (torch.Tensor): A tensor of shape (H*W) containing the common elements for the neuron.
        unique_elements (torch.Tensor): A tensor of shape (H*W) containing the unique elements for the neuron.
        uncoverable_elements (torch.Tensor): A tensor of shape (H*W) containing the uncoverable elements for the neuron.
    Returns:
        tuple: A tuple containing the common intersection, unique intersection, common extras, unique extras, and uncoverable quantities for the neuron.
    """
    common_elements = common_elements.to(bitmaps.device)
    unique_elements = unique_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)

    # Sample
    bitmaps_unique = (bitmaps & unique_elements).sum(dim=1)
    bitmaps_common = (bitmaps & common_elements).sum(dim=1)
    bitmaps_coverable = bitmaps & (~uncoverable_elements)
    bitmaps_sum = bitmaps.sum(dim=1)
    bitmaps_coverable_sum = bitmaps_coverable.sum(dim=1)
    space_extras = ~bitmaps
    common_space_extras = (space_extras & common_elements).sum(dim=1)
    unique_space_extras = (space_extras & unique_elements).sum(dim=1)

    # Sum
    bitmaps_unique_sum = np.int64(bitmaps_unique.sum().item())
    bitmaps_common_sum = np.int64(bitmaps_common.sum().item())
    num_hits = np.int64(bitmaps_sum.sum().item())
    coverable_hits = np.int64(bitmaps_coverable_sum.sum().item())
    common_extras_sum = np.int64(common_space_extras.sum().item())
    unique_extras_sum = np.int64(unique_space_extras.sum().item())

    # Move to CPU and convert to numpy
    bitmaps_unique = bitmaps_unique.cpu().numpy()
    bitmaps_common = bitmaps_common.cpu().numpy()
    bitmaps_sum = bitmaps_sum.cpu().numpy()
    bitmaps_coverable_sum = bitmaps_coverable_sum.cpu().numpy()
    common_space_extras = common_space_extras.cpu().numpy()
    unique_space_extras = unique_space_extras.cpu().numpy()
    bitmaps_coverable = bitmaps_coverable.cpu()
    return (
        (bitmaps_unique, bitmaps_unique_sum),
        (bitmaps_common, bitmaps_common_sum),
        (bitmaps_coverable_sum, coverable_hits),
        (bitmaps_sum, num_hits),
        (common_space_extras, common_extras_sum),
        (unique_space_extras, unique_extras_sum),
    )


def get_concept_intersection_quantities(
    *, bitmaps, masks, masks_quantities, neuron_quantities
):
    """Gets the concept intersection quantities for each concept in the masks.
    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        masks (list): A list of concept masks. Each mask is a tensor of shape (H*W).
        masks_quantities (tuple): A tuple containing the common intersection, unique intersection, common extras, unique extras, and uncoverable quantities for each mask.
        neuron_quantities (tuple): A tuple containing the common intersection, unique intersection, common extras, unique extras, and uncoverable quantities for the neuron.

    Returns:
        list: A list of tuples containing the concept quantities for each concept in the masks.
    """
    candidate_concepts = range(len(masks))
    num_concepts = len(masks)

    common_elements, unique_elements, _ = masks_quantities
    (bitmaps_unique, _), (bitmaps_common, _), _, _, _, _ = neuron_quantities

    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    concepts_quantities = [0] * num_concepts

    for concept in candidate_concepts:
        concept_mask = general_utils.parse_mask_by_type(masks[concept])
        concept_quantities = compute_quantities_vector(
            label_mask=concept_mask,
            bitmaps=bitmaps,
            common_elements=common_elements,
            unique_elements=unique_elements,
            neuron_common=bitmaps_common,
            neuron_unique=bitmaps_unique,
        )
        concept_info = get_concept_info(concept_quantities)
        concepts_quantities[concept] = concept_info

    return concepts_quantities


def get_optimal_heuristic_info(*, masks, bitmaps, masks_quantities):
    """Gets the optimal heuristic info for the beam optimal search.
    Args:
        masks (list): A list of concept masks. Each mask is a tensor of
            shape (N, H*W).
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        masks_quantities (tuple): A tuple containing the common intersection, unique intersection, common extras, unique extras, and uncoverable quantities for each mask.
    Returns:
        tuple: A tuple containing the neuron quantities and the concept quantities.
    """

    common_elements, unique_elements, uncoverable_elements = masks_quantities
    neuron_quantities = get_neuron_quantities(
        bitmaps=bitmaps,
        common_elements=common_elements,
        unique_elements=unique_elements,
        uncoverable_elements=uncoverable_elements,
    )
    concepts_quantities = get_concept_intersection_quantities(
        bitmaps=bitmaps,
        masks=masks,
        masks_quantities=masks_quantities,
        neuron_quantities=neuron_quantities,
    )
    return neuron_quantities, concepts_quantities


def estimate_min_max_iou_from_label_info(
    *, label_quantities, neuron_quantities, num_hits, minimum_threshold=0
):
    """Compute the IoU estimation for a given label based on its quantities.
    Args:
        label_quantities (dict): A dictionary containing the quantities for the label.
        neuron_quantities (tuple): A tuple containing the neuron quantities.
        num_hits (int): The number of hits for the label.
        minimum_threshold (float): The minimum threshold for the estimation. If the estimation is below this threshold, it will be set to 0.
    Returns:
        tuple: The maximum and minimum estimation for the label.
    """

    max_iou = compute_max_iou_from_label_info(
        label_quantities, neuron_quantities, num_hits
    )
    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0

    # Min IoU
    min_iou = compute_min_iou_from_label_info(label_quantities, num_hits)

    return max_iou, min_iou


def compute_min_iou_from_label_info(label_quantities, num_hits):
    """Compute the minimum IoU estimation for a given label based on its quantities.
    Args:
        label_quantities (dict): A dictionary containing the quantities for the label.
        num_hits (int): The number of hits for the label.
    Returns:
        tuple: The maximum and minimum estimation for the label.
    """

    # Unpack max and min quantities
    # Note that we use the sum for all quantities. But the way in which the sum is computed depends on the specific heuristic used to compute
    # the label quantites. The sum heuristic simply aggregates the sum and thus results in a larger estimation. Coversely, the sample heuristic
    # compute the sum over quantities aggregated per sample, which results in a smaller estimation. The optimal heuristic computes the sum over quantities aggregated per sample, but it also computes the sum over the sums of the quantities, which results in a larger estimation.
    min_common_intersection_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    unique_intersection_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="unique_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    max_common_extras_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    unique_extras_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )

    # Min IoU
    min_intersection = min_common_intersection_sum + unique_intersection_sum
    max_union = num_hits + unique_extras_sum + max_common_extras_sum
    min_iou = min_intersection / max_union

    return min_iou


def compute_max_iou_from_label_info(label_quantities, neuron_quantities, num_hits):
    """Computes the maximum IoU estimation for a given label based on its quantities.
    Args:
        label_quantities (dict): A dictionary containing the quantities for the label.
        neuron_quantities (tuple): A tuple containing the neuron quantities.
        num_hits (int): The number of hits for the neuron.
    Returns:
    float: The IoU for the label.
    """
    max_common_intersection_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    unique_intersection_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_common_extras_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sum",
    )
    unique_extras_sum = get_quantity(
        label_info=label_quantities,
        quantity_name="unique_extras",
        quantity_type="min",
        quantity_scope="sum",
    )
    _, _, (_, neuron_coverable_sum), _, _, _ = neuron_quantities

    max_label_common_intersection_sum = min(
        max_common_intersection_sum + unique_intersection_sum, neuron_coverable_sum
    ).item()
    if max_label_common_intersection_sum == 0:
        return 0.0
    label_iou = (
        max_label_common_intersection_sum
        / (num_hits + min_common_extras_sum + unique_extras_sum).item()
    )
    return label_iou


def compute_label_info_and_iou_from_mask(
    label_mask, *, bitmaps, heuristic_info, num_hits
):
    """Compute the quantities, the info and the IoU of a label given its mask.

    Args:
        label_mask (torch.Tensor): A tensor of shape (H*W) representing the mask of the label to compute the info and IoU for.
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        num_hits (int): The number of hits of the bitmap.

    Returns:
        label_info (dict): A dictionary containing the quantities for the label. The keys of the dictionary are the names of the quantities and the values are the corresponding values for the label.
        label_iou (float): The IoU of the label with the bitmap.
    """
    seg_quantities, neuron_quantities, _ = heuristic_info
    (neuron_unique, _), (neuron_common, _), _, _, _, _ = neuron_quantities
    common_elements, unique_elements, uncoverable_elements = seg_quantities
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)
    label_quantities = compute_quantities_vector(
        label_mask=label_mask,
        bitmaps=bitmaps,
        common_elements=common_elements,
        unique_elements=unique_elements,
        neuron_common=neuron_common,
        neuron_unique=neuron_unique,
    )
    # Compute the info and iou from the quantities
    label_info = get_concept_info(label_quantities)
    label_iou = compute_max_iou_from_label_info(
        label_info, neuron_quantities, num_hits=num_hits
    )
    return label_info, label_iou


def update_heuristic_info(*, nodes_info, heuristic_info, label_mapping, max_length):
    """Update the heuristic info with the new nodes info.
    Args:
        nodes_info (dict): A dictionary containing the quantities for the new nodes to add to the heuristic info. The keys of the dictionary are the nodes and the values are the corresponding quantities for each node.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        label_mapping (dict): A dictionary mapping formulas to their index in the heuristic info. This is used to update the concept quantities in the heuristic info with the new nodes info.
        max_length (int): The maximum length of the labels to consider for the heuristic info. If a label is longer than this length, it will not be added to the heuristic info to save memory.

    Returns:
        tuple: A tuple containing the updated heuristic info and the updated label mapping.
    """
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info

    for node, node_quantities in nodes_info.items():
        if len(node) >= max_length:
            # In this case the information is not useful, we skip it to save memory
            continue
        if isinstance(node, F.Not) and node.val in label_mapping:
            # We can already compute its quantities
            continue
        elif node not in label_mapping:
            label_mapping[node] = len(label_mapping)
            concepts_quantities.append(node_quantities)
        else:
            index_node = label_mapping[node]
            # If the node is already present, we update the concept quantities
            concepts_quantities[index_node] = node_quantities

    updated_heuristic_info = (seg_quantities, neuron_quantities, concepts_quantities)
    return updated_heuristic_info, label_mapping
