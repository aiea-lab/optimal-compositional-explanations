"""Aggregate-sum optimal heuristic estimators for compositional formulas."""

import numpy as np

from utils import optimal_utils
from compositional import formula as F
from utils.constants import TOP_INDEX_SUM, BOTTOM_INDEX_SUM


def estimate_disjoint_label_info(label, *, left_quantities, right_quantities):
    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """

    if not isinstance(label, F.Or):
        # We cover two cases here:
        # 1) AND of disjoint labels is zero by definition
        # 2) AND NOT of disjoint labels is zero by definition and unverified.
        # Since they are disjoint, there is not a counter example of their presence together
        # Everything would end up the same of the left label. This is the case where part of the explanation would be unverified

        return None

    # Left quantities
    left_unique_intersection_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    max_left_common_intersection_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_left_common_intersection_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    max_left_common_extras_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_left_common_extras_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sum",
    )
    left_unique_extras_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )

    # Right quantities
    right_unique_intersection_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )

    max_right_common_intersection_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_right_common_intersection_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    max_right_common_extras_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_right_common_extras_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sum",
    )
    right_unique_extras_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )


    # Since they are disjoint and the OR is additive, everything is just the sum of the two sides
    unique_intersection_sum = (
        left_unique_intersection_sum + right_unique_intersection_sum
    )
    max_common_intersection_sum = (
        max_left_common_intersection_sum + max_right_common_intersection_sum
    )
    min_common_intersection_sum = (
        min_left_common_intersection_sum + min_right_common_intersection_sum
    )
    if (
        (unique_intersection_sum <= left_unique_intersection_sum)
        and (max_common_intersection_sum <= max_left_common_intersection_sum)
        and (min_common_intersection_sum <= min_left_common_intersection_sum)
    ) or (
        (unique_intersection_sum <= right_unique_intersection_sum)
        and (max_common_intersection_sum <= max_right_common_intersection_sum)
        and (min_common_intersection_sum <= min_right_common_intersection_sum)
    ):
        # If one of the two side does not change to the intersection, we can discard this formula
        return None

    unique_extras_sum = left_unique_extras_sum + right_unique_extras_sum

    min_common_extras_sum = min_left_common_extras_sum + min_right_common_extras_sum
    max_common_extras_sum = max_left_common_extras_sum + max_right_common_extras_sum

    min_unique_extras_sum = unique_extras_sum
    return (
        ((None, max_common_intersection_sum), (None, min_common_intersection_sum)),
        (None, unique_intersection_sum),
        ((None, max_common_extras_sum), (None, min_common_extras_sum)),
        (None, min_unique_extras_sum),
        ((None, None), (None, None)),
        ((None, None), (None, None)),
    )


def estimate_label_info(label, *, left_quantities, right_quantities, neuron_quantities):
    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.
        neuron_quantities (tuple): Quantities from the neuron.

    Returns:
        tuple: Estimated quantities for the label.
    """

    _, neuron_common_tuple, _, _, common_space_extras_tuple, _ = neuron_quantities
    _, neuron_common_sum = neuron_common_tuple
    _, common_space_extras_sum = common_space_extras_tuple

    # Left quantities
    left_unique_intersection_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )

    max_left_common_intersection_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_left_common_intersection_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    max_left_common_extras_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_left_common_extras_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sum",
    )
    left_unique_extras_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )

    # Right quantities
    right_unique_intersection_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    max_right_common_intersection_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_right_common_intersection_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    max_right_common_extras_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_right_common_extras_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sum",
    )
    right_unique_extras_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    max_right_common_uncovered_sum = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_uncovered",
        quantity_type="max",
        quantity_scope="sum",
    )

    if isinstance(label, F.Or):

        # Unique elements are additive since they are disjoint
        unique_intersection_sum = left_unique_intersection_sum + right_unique_intersection_sum
        unique_extras_sum = left_unique_extras_sum + right_unique_extras_sum

        # Worst case scenario is assuming they are fully overlapping in the intersection
        min_common_intersection_sum = max(
            min_left_common_intersection_sum, min_right_common_intersection_sum
        )

        # Best case scenario is assuming they are disjoint in the intersection, but they cannot be more than the neuron common sum
        max_common_intersection_sum = min(
            max_left_common_intersection_sum + max_right_common_intersection_sum,
            neuron_common_sum,
        )

        if (
            unique_intersection_sum <= left_unique_intersection_sum
            and max_common_intersection_sum <= max_left_common_intersection_sum
            and min_common_intersection_sum <= min_left_common_intersection_sum
        ) or (
            unique_intersection_sum <= right_unique_intersection_sum
            and max_common_intersection_sum <= max_right_common_intersection_sum
            and min_common_intersection_sum <= min_right_common_intersection_sum
        ):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None

        # Same reasoning for the extras, but we need to consider the common space extras that can be added to both sides
        min_common_extras_sum = max(
            min_left_common_extras_sum, min_right_common_extras_sum
        )
        max_common_extras_sum = min(
            common_space_extras_sum,
            max_left_common_extras_sum + max_right_common_extras_sum,
        )  # min(max_size_mask - N^u - N^c - E^u(L) - E^u(c), E_max^c(L) + E^c(c))

    elif isinstance(label, F.And) and isinstance(label.right, F.Not):

        # The AND NOT preserves all the unique elements of the left side, since the NOT cannot have overlap there (because we are considering unique elements)
        unique_intersection_sum = left_unique_intersection_sum
        unique_extras_sum = left_unique_extras_sum

        # Common Elements. AND NOT cannot increase common elements
        # Best case scenario is when they were disjoint in the common part. The negation would preserve all the common elements of the left side in this case
        max_common_intersection_sum = min(
            max_right_common_uncovered_sum, max_left_common_intersection_sum
        )  # min(U_max^c, I_max^c(L))
        # Worst case scenario is when they were fully overlapping in the common part. The negation would remove all the common elements of the left side in this case
        min_common_intersection_sum = max(
            min_left_common_intersection_sum - max_right_common_intersection_sum, 0
        )

        # Same reasoning as for the common intersection
        min_common_extras_sum = max(
            min_left_common_extras_sum - max_right_common_extras_sum, 0
        )
        max_common_extras_sum = min(
            max_left_common_extras_sum,
            common_space_extras_sum - min_right_common_extras_sum,
        )  # E_max^c(L) - max(0, E_min^c(L) + E_min^c(c) - E_max^c(c))

        if max_left_common_extras_sum == 0 and min_left_common_extras_sum == 0:
            return None

    elif isinstance(label, F.And):
        # The AND always zeroes the unique elements
        unique_extras_sum = 0
        unique_extras_sum = 0
        unique_intersection_sum = 0
        unique_intersection_sum = 0

        # Common Elements. AND  cannot increase common elements
        # Best case scenario is when they were fully overlapping in the common part. The intersection would preserve all the common elements in this case
        max_common_extras_sum = min(
            max_left_common_extras_sum, max_right_common_extras_sum
        )
        # Worst case scenario is when they were disjoint in the common part. The intersection would remove all the common elements in this case.
        # THe inclusion for the subtraction against common space extras is because there could be a "mandatory" overlap if the two terms are big enough
        min_common_extras_sum = max(
            min_left_common_extras_sum
            + min_right_common_extras_sum
            - common_space_extras_sum,
            0,
        )

        if (
            left_unique_extras_sum == 0
            and max_left_common_extras_sum == 0
            and min_left_common_extras_sum == 0
        ):
            return None

        # Same reasoning as for the common extras
        max_common_intersection_sum = min(
            max_left_common_intersection_sum, max_right_common_intersection_sum
        )
        min_common_intersection_sum = max(
            min_left_common_intersection_sum
            + min_right_common_intersection_sum
            - neuron_common_sum,
            0,
        )

    return (
        ((None, max_common_intersection_sum), (None, min_common_intersection_sum)),
        (None, unique_intersection_sum),
        ((None, max_common_extras_sum), (None, min_common_extras_sum)),
        (None, unique_extras_sum),
        ((None, None), (None, None)),
        ((None, None), (None, None)),
    )


def or_chain_estimation(
    label,
    *,
    label_quantities,
    neuron_quantities,
    max_improvement,
    num_hits,
    max_size_mask,
    max_length
):
    """Estimate the max and min IoU of an OR chain starting from the given label, based on the current quantities and the potential improvement.

    Args:
        label (F.Formula): The label for which to estimate the IoU.
        label_quantities (tuple): Quantities from the label.
        neuron_quantities (tuple): Quantities from the neuron.
        max_improvement (tuple): The potential improvement for a chain up to max_length.
        num_hits (int): The number of hits.
        max_size_mask (int): The maximum size of the mask.
        max_length (int): The maximum length of the chain.

    Returns:
        tuple: Estimated max and min IoU for the OR chain.
    """

    # Unpack max and min quantities
    max_common_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_common_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    unique_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )

    max_common_extras_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_common_extras_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sum",
    )
    unique_extras_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + unique_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Compute potential improvement
    k = max_length - len(label)  # Missing length
    (
        improv_common_intersection,
        improv_unique_intersection,
        improv_common_extras,
        improv_unique_extras,
        _,
        _,
    ) = max_improvement
    top_k_common_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM]
    top_k_unique_intersection_sum = improv_unique_intersection[k][TOP_INDEX_SUM]
    top_k_common_extras_sum = improv_common_extras[k][TOP_INDEX_SUM]
    top_k_unique_extras_sum = improv_unique_extras[k][TOP_INDEX_SUM]
    bottom_1_common_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM]
    bottom_1_unique_extras_sum = improv_unique_extras[0][BOTTOM_INDEX_SUM]

    # Aux variables
    (
        neuron_unique_tuple,
        neuron_common_tuple,
        neuron_coverable_tuple,
        _,
        common_space_extras_tuple,
        unique_space_extras_tuple,
    ) = neuron_quantities
    _, unique_space_extras_sum = unique_space_extras_tuple
    _, common_space_extras_sum = common_space_extras_tuple
    _, neuron_common_sum = neuron_common_tuple
    _, neuron_unique_sum = neuron_unique_tuple
    neuron_coverable, neuron_coverable_sum = neuron_coverable_tuple
    neuron_coverable_sum = neuron_coverable_tuple[1]
    bottom_1_intersection_sum = (
        improv_common_intersection[0][BOTTOM_INDEX_SUM]
        + improv_unique_intersection[0][BOTTOM_INDEX_SUM]
    )
    tot_size = np.int64(max_size_mask * len(neuron_coverable))

    # Max IoU Quantitites
    min_union = num_hits + max(
        min_common_extras_sum + unique_extras_sum,
        bottom_1_common_extras_sum + bottom_1_unique_extras_sum,
    )
    max_intersection = min(
        min(
            max_common_intersection_sum + top_k_common_intersection_sum,
            neuron_common_sum,
        )
        + min(
            unique_intersection_sum + top_k_unique_intersection_sum,
            neuron_unique_sum,
        ),
        neuron_coverable_sum,
    )

    # Min IoU Quantities
    min_intersection = max(
        min(
            min_common_intersection_sum + unique_intersection_sum,
            neuron_coverable_sum,
        ),
        bottom_1_intersection_sum,
    )
    max_union = min(
        num_hits
        + min(max_common_extras_sum + top_k_common_extras_sum, common_space_extras_sum)
        + min(unique_extras_sum + top_k_unique_extras_sum, unique_space_extras_sum),
        tot_size,
    )

    return (max_intersection, min_intersection), (max_union, min_union)


def and_chain_estimation(
    *, label_quantities, neuron_quantities, max_improvement, num_hits, max_size_mask
):
    """Estimate the max and min IoU of an AND chain starting from the given label, based on the current quantities and the potential improvement.

    Args:
        label_quantities (tuple): Quantities from the label.
        neuron_quantities (tuple): Quantities from the neuron.
        max_improvement (tuple): The potential improvement for a chain up to max_length.
        num_hits (int): The number of hits.
        max_size_mask (int): The maximum size of the mask.

    Returns:
        tuple: Estimated max and min IoU for the AND chain.

    """
    max_common_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    min_common_extras_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sum",
    )

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Aux variables
    _, _, neuron_coverable_tuple, _, common_space_extras_tuple, _ = neuron_quantities
    neuron_coverable, _ = neuron_coverable_tuple
    _, common_space_extras_sum = common_space_extras_tuple
    neuron_coverable = neuron_coverable_tuple[0]
    max_common_extras_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )

    # Compute potential improvement
    improv_common_intersection, _, improv_common_extras, _, _, _ = max_improvement
    top_1_common_intersection_sum = improv_common_intersection[0][TOP_INDEX_SUM]
    bottom_1_common_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM]
    top_1_common_extras_sum = improv_common_extras[0][TOP_INDEX_SUM]
    tot_size = np.int64(max_size_mask * len(neuron_coverable))

    # MaxIoU
    max_intersection = min(max_common_intersection_sum, top_1_common_intersection_sum)
    min_union = min(
        num_hits
        + max(
            np.int64(0),
            min_common_extras_sum
            + bottom_1_common_extras_sum
            - common_space_extras_sum,
        ),
        tot_size,
    )

    # Min Iou
    min_intersection = np.int64(0)
    max_union = num_hits + min(max_common_extras_sum, top_1_common_extras_sum)

    return (max_intersection, min_intersection), (max_union, min_union)


def and_not_chain_estimation(
    *, label_quantities, neuron_quantities, max_improvement, num_hits, max_size_mask
):
    """Estimate the max and min IoU of an AND NOT chain starting from the given label, based on the current quantities and the potential improvement.

    Args:
        label_quantities (tuple): Quantities from the label.
        neuron_quantities (tuple): Quantities from the neuron.
        max_improvement (tuple): The potential improvement for a chain up to max_length.
        num_hits (int): The number of hits.
        max_size_mask (int): The maximum size of the mask.

    Returns:
        tuple: Estimated max and min IoU for the AND NOT chain.
    """
    # Unpack max and min quantities
    max_common_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    unique_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )


    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + unique_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Aux variables
    _, _, improv_common_extras, _, _, _ = max_improvement
    _, _, neuron_coverable_tuple, _, common_space_extras_tuple, _ = neuron_quantities
    neuron_coverable, neuron_coverable_sum = neuron_coverable_tuple
    tot_size = np.int64(max_size_mask * len(neuron_coverable))
    _, common_space_extras_sum = common_space_extras_tuple

    max_common_extras_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sum",
    )
    unique_extras_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )

    # Max IoU
    max_intersection = min(
        unique_intersection_sum + max_common_intersection_sum, neuron_coverable_sum
    )
    min_union = num_hits + unique_extras_sum

    # Min IoU
    bottom_1_common_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM]
    max_improv_common_extras_sum = min(
        max_common_extras_sum, common_space_extras_sum - bottom_1_common_extras_sum
    )
    min_intersection = unique_intersection_sum
    max_union = min(
        num_hits + unique_extras_sum + max_improv_common_extras_sum, tot_size
    )

    return (max_intersection, min_intersection), (max_union, min_union)
