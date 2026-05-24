"""Sample-based heuristic estimations for compositional formulas.
"""

import numpy as np

from utils import optimal_utils
from compositional import formula as F
from utils.constants import (
    TOP_INDEX_SAMPLE,
    BOTTOM_INDEX_SAMPLE,
    BOTTOM_INDEX_SUM,
    TOP_INDEX_SUM,
)


########### DISJOINT CASE ###########
def can_improve_or_iou_disjoint_case(left_quantities, right_quantities):
    """Check if it is possible to improve the IoU estimation in the disjoint case based on the left and right quantities.
    Args:
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.
    Returns:
        bool: True if it is possible to improve the IoU estimation, False otherwise.
    """
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

    # If one of the sides has zero intersection sums, we set the estimation to None since one of the two terms does not contribute
    # Indeed AND and NOT are 0 for disjint (NOT because it would be unverified) and for OR, since the operator is additive,
    # the result would be only and increases of extras without gain in the intersection. Therefore, any formula obtainable by
    # (A OR B) it is guranteed to be <= the one obtainable by only A or only B (Based on which one doesn't have intersection ==0)
    if (
        right_unique_intersection_sum == 0
        and max_right_common_intersection_sum == 0
        and min_right_common_intersection_sum == 0
    ) or (
        left_unique_intersection_sum == 0
        and max_left_common_intersection_sum == 0
        and min_left_common_intersection_sum == 0
    ):
        return False
    else:
        return True


def estimate_disjoint_label_info(label, left_quantities, right_quantities):
    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """

    if not isinstance(label, F.Or) or not can_improve_or_iou_disjoint_case(
        left_quantities, right_quantities
    ):
        # We cover three cases here:
        # 1) AND of disjoint labels is zero by definition
        # 2) AND NOT of disjoint labels is zero by definition and unverified.
        # Since they are disjoint, there is not a counter example of their presence together
        # Everything would end up the same of the left label. This is the case where part of the explanation would be unverified
        # 3) For Or, The label can be discarded if it cannot improve the iou of the best single label

        return None

    # OR CASE
    max_left_common_intersection = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_left_common_intersection = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sample",
    )
    left_unique_intersection = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    max_left_common_extras = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_left_common_extras = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sample",
    )
    left_unique_extras = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sample",
    )
    max_right_common_intersection = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_right_common_intersection = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sample",
    )
    right_unique_intersection = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    max_right_common_extras = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_right_common_extras = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sample",
    )
    right_unique_extras = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sample",
    )

    unique_intersection = (
        left_unique_intersection + right_unique_intersection
    )  # I^u(L) + I^u(c)
    max_common_intersection = (
        max_left_common_intersection + max_right_common_intersection
    )  # I_max^c(L) + I_max^c(c)
    min_common_intersection = (
        min_left_common_intersection + min_right_common_intersection
    )  # I_min^c(L) + I_min^c(c)
    unique_extras = (
        left_unique_extras + right_unique_extras
    )  # E^u(L) + E^u(c)
    min_common_extras = (
        min_left_common_extras + min_right_common_extras
    )  # E_min^c(L) + E_min^c(c)
    max_common_extras = (
        max_left_common_extras + max_right_common_extras
    )  # E_max^c(L) + E_max^c(c)

    return (
        (
            (max_common_intersection, max_common_intersection.sum()),
            (min_common_intersection, min_common_intersection.sum()),
        ),
            (unique_intersection, unique_intersection.sum()),
        (
            (max_common_extras, max_common_extras.sum()),
            (min_common_extras, min_common_extras.sum()),
        ),
            (unique_extras, unique_extras.sum()),
        ((None, None), (None, None)),
        ((None, None), (None, None)),
    )


############ OTHER CASES ###########
def estimate_label_info(label, left_quantities, right_quantities, neuron_quantities):
    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """

    # Quantities used by all the operators
    _, neuron_common_tuple, _, _, common_space_extras_tuple, _ = neuron_quantities
    neuron_common, _ = neuron_common_tuple
    common_space_extras, _ = common_space_extras_tuple
    max_left_common_intersection = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_left_common_intersection = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sample",
    )
    left_unique_intersection = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    max_left_common_extras = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_left_common_extras = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sample",
    )
    left_unique_extras = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sample",
    )


    # Sum of the left quantities
    left_unique_intersection_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    left_unique_extras_sum = optimal_utils.get_quantity(
        label_info=left_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sum",
    )

    max_right_common_intersection = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_right_common_intersection = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sample",
    )

    max_right_common_extras = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_right_common_extras = optimal_utils.get_quantity(
        label_info=right_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sample",
    )

    if isinstance(label, F.Or):
        # Auxiliary info: Sum of the right quantities, right unique extras and intersection
        right_unique_intersection_sum = optimal_utils.get_quantity(
            label_info=right_quantities,
            quantity_name="unique_intersection",
            quantity_type="max",
            quantity_scope="sum",
        )
        right_unique_extras_sum = optimal_utils.get_quantity(
            label_info=right_quantities,
            quantity_name="unique_extras",
            quantity_type="max",
            quantity_scope="sum",
        )
        right_unique_intersection = optimal_utils.get_quantity(
            label_info=right_quantities,
            quantity_name="unique_intersection",
            quantity_type="max",
            quantity_scope="sample",
        )
        right_unique_extras = optimal_utils.get_quantity(
            label_info=right_quantities,
            quantity_name="unique_extras",
            quantity_type="max",
            quantity_scope="sample",
        )

        # OR simply sums the unique elements
        unique_intersection = left_unique_intersection + right_unique_intersection # I^u(L) + I^u(c)

        unique_intersection_sum = left_unique_intersection_sum + right_unique_intersection_sum
        

        # OR is additive. Therefore the minum can't be lower than the maximum minimum of the two sides
        min_common_intersection = np.maximum(
            min_left_common_intersection, min_right_common_intersection
        )  # max(I_min^c(L), I_min^c(c))

        # The best case scenario is when the two sides are disjoint. In that case it is simply the sum
        max_common_intersection = np.minimum(
            max_left_common_intersection + max_right_common_intersection, neuron_common
        )

        if (
            unique_intersection_sum <= left_unique_intersection_sum
            and np.all(max_common_intersection <= max_left_common_intersection)
            and np.all(min_common_intersection <= min_left_common_intersection)
        ) or (
            unique_intersection_sum <= right_unique_intersection_sum
            and np.all(max_common_intersection <= max_right_common_intersection)
            and np.all(min_common_intersection <= min_right_common_intersection)
        ):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None

        # OR simply sums the unique elements
        unique_extras = left_unique_extras + right_unique_extras # E^u(L) + E^u(c)
        unique_extras_sum = left_unique_extras_sum + right_unique_extras_sum

        # Same reasoning as for the intersection
        min_common_extras = np.maximum(
            min_left_common_extras, min_right_common_extras
        )  # max(E_min^c(L), E_min^c(c))
        max_common_extras = np.minimum(
            max_left_common_extras + max_right_common_extras, common_space_extras
        )  # min(max_size_mask - N^u - N^c - E^u(L) - E^u(c), E_max^c(L) + E_max^c(c))

    elif isinstance(label, F.And) and isinstance(label.right, F.Not):
        # Auxiliary info for AND NOT: uncovered elements of the right label
        max_right_common_uncovered = optimal_utils.get_quantity(
            label_info=right_quantities,
            quantity_name="common_uncovered",
            quantity_type="max",
            quantity_scope="sample",
        )
        min_right_common_uncovered = optimal_utils.get_quantity(
            label_info=right_quantities,
            quantity_name="common_uncovered",
            quantity_type="min",
            quantity_scope="sample",
        )

        # The AND NOT preserves all the left unique elements and none of the right unique elements
        unique_intersection = left_unique_intersection
        unique_intersection_sum = left_unique_intersection_sum
        unique_extras = left_unique_extras 
        unique_extras_sum = left_unique_extras_sum

        # Common elements estimation
        min_common_intersection = np.clip(
            min_left_common_intersection + min_right_common_uncovered - neuron_common,
            a_min=0,
            a_max=None,
        )  # max(0, I_min^c(L) + U_min^c - N^u - N^c)

        max_common_intersection = np.minimum(
            max_right_common_uncovered, max_left_common_intersection
        )  # min(U_max^c, I_max^c(L))

        min_common_extras = np.clip(
            min_left_common_extras - max_right_common_extras, a_min=0, a_max=None
        )

        max_common_extras = np.minimum(
            max_left_common_extras, common_space_extras - min_right_common_extras
        )

    elif isinstance(label, F.And):
        # AND cannot preserve unique elements. Therefore we zero them out
        unique_extras = np.zeros_like(left_unique_extras)
        unique_intersection = np.zeros_like(left_unique_intersection)
        unique_intersection_sum = 0
        unique_extras_sum = 0

        # Common Elements
        min_common_extras = np.clip(
            min_left_common_extras + min_right_common_extras - common_space_extras,
            a_min=0,
            a_max=None,
        )  # max(0, E^c(L) + E^c(c) - (max_size_mask - N^u - N^c))

        max_common_extras = np.minimum(
            max_left_common_extras, max_right_common_extras
        )  # min(E^c(L), E^c(c))

        min_common_intersection = np.clip(
            min_left_common_intersection
            + min_right_common_intersection
            - neuron_common,
            a_min=0,
            a_max=None,
        )  # max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)

        max_common_intersection = np.minimum(
            max_left_common_intersection, max_right_common_intersection
        )  # min(I_max^c(L), I_max^c(c))

    else:
        raise ValueError(f"Unknown label type: {type(label)}")

    return (
        (
            (max_common_intersection, max_common_intersection.sum()),
            (min_common_intersection, min_common_intersection.sum()),
        ),
            (unique_intersection, unique_intersection_sum),
        (
            (max_common_extras, max_common_extras.sum()),
            (min_common_extras, min_common_extras.sum()),
        ),
            (unique_extras, unique_extras_sum),
        ((None, None), (None, None)),
        ((None, None), (None, None)),
    )


############# HEURISTIC ESTIMATION FOR PATHS #############
def or_chain_estimation(
    label,
    *,
    label_quantities,
    neuron_quantities,
    max_improvement,
    num_hits,
    max_size_mask,
    max_length,
):
    """Estimate the max and min IoU for an OR chain based on the label quantities, neuron quantities and improvement information.
    Args:
        label (F.Formula): The label for which to estimate the IoU.
        label_quantities (tuple): Quantities for the label.
        neuron_quantities (tuple): Quantities for the neuron.
        max_improvement (tuple): Information about the maximum improvement.
        num_hits (int): Number of hits for the current formula.
        max_size_mask (int): The maximum size of the mask.
        max_length (int): The maximum length of the formula.
    Returns:
        tuple: Estimated max and min IoU for the OR chain.
    """
    # Sum quantities
    unique_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )
    max_common_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + unique_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Unpack quantities
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
    min_common_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sum",
    )
    max_common_intersection = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_common_intersection = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="min",
        quantity_scope="sample",
    )
    unique_intersection = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    max_common_extras = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sample",
    )
    min_common_extras = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="min",
        quantity_scope="sample",
    )
    unique_extras = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_extras",
        quantity_type="max",
        quantity_scope="sample",
    )


    (
        improv_common_intersection,
        improv_unique_intersection,
        improv_common_extras,
        improv_unique_extras,
        _,
        _,
    ) = max_improvement
    (
        neuron_unique_tuple,
        neuron_common_tuple,
        neuron_coverable_tuple,
        neuron_sum_tuple,
        common_space_extras_tuple,
        unique_space_extras_tuple,
    ) = neuron_quantities
    common_space_extras, _ = common_space_extras_tuple
    unique_space_extras, _ = unique_space_extras_tuple
    neuron_coverable, _ = neuron_coverable_tuple
    neuron_common, _ = neuron_common_tuple
    neuron_unique, _ = neuron_unique_tuple

    # Aux variables
    tot_size = max_size_mask * len(neuron_coverable)
    label_len = len(label)
    k = max_length - label_len  # Missing length

    # Max IoU Quantitites
    neuron_sum = neuron_sum_tuple[0]
    top_k_common_intersection = improv_common_intersection[k][TOP_INDEX_SAMPLE]
    top_k_unique_intersection = improv_unique_intersection[k][TOP_INDEX_SAMPLE]
    max_intersection = (
        np.minimum(max_common_intersection + top_k_common_intersection, neuron_common)
        + np.minimum(unique_intersection + top_k_unique_intersection, neuron_unique)
    ).sum()

    bottom_1_extras_sum = (
        improv_common_extras[0][BOTTOM_INDEX_SUM]
        + improv_unique_extras[0][BOTTOM_INDEX_SUM]
    )
    if bottom_1_extras_sum > 0:
        bottom_1_extras_common = improv_common_extras[0][BOTTOM_INDEX_SAMPLE]
        bottom_1_extras_unique = improv_unique_extras[0][BOTTOM_INDEX_SAMPLE]
        min_label_extras = min_common_extras + unique_extras
        min_extras = np.maximum(
            min_label_extras, bottom_1_extras_common + bottom_1_extras_unique
        )
        min_union = np.clip(neuron_sum + min_extras, a_min=0, a_max=max_size_mask).sum()
    else:
        min_union = min(
            num_hits + min_common_extras_sum + unique_extras_sum, tot_size
        )

    # Min IoU Quantitites
    bottom_1_intersection_sum = (
        improv_common_intersection[0][BOTTOM_INDEX_SUM]
        + improv_unique_intersection[0][BOTTOM_INDEX_SUM]
    )
    if bottom_1_intersection_sum > 0:
        bottom_1_intersection = (
            improv_common_intersection[0][BOTTOM_INDEX_SAMPLE]
            + improv_unique_intersection[0][BOTTOM_INDEX_SAMPLE]
        )
        min_intersection = np.maximum(
            min_common_intersection + unique_intersection, bottom_1_intersection
        ).sum()
    else:
        min_intersection = min_common_intersection_sum + unique_intersection_sum

    top_k_common_extras = improv_common_extras[k][TOP_INDEX_SAMPLE]
    top_k_unique_extras = improv_unique_extras[k][TOP_INDEX_SAMPLE]
    max_union = np.clip(
        neuron_sum
        + np.minimum(max_common_extras + top_k_common_extras, common_space_extras)
        + np.minimum(unique_extras + top_k_unique_extras, unique_space_extras),
        a_min=0,
        a_max=max_size_mask,
    ).sum()

    return (max_intersection, min_intersection), (max_union, min_union)


def and_chain_estimation(
    *,
    label_quantities,
    neuron_quantities,
    max_improvement,
    num_hits,
    max_size_mask=None,
):
    """Estimate the max and min IoU for an AND chain based on the label quantities, neuron quantities and improvement information.
    Args:
        label_quantities (tuple): Quantities for the label.
        neuron_quantities (tuple): Quantities for the neuron.
        max_improvement (tuple): Information about the maximum improvement.
        num_hits (int): Number of hits for the current formula.
        max_size_mask (int): The maximum size of the mask. Not used in the sample heuristic but we keep it for consistency with the sum variant
    Returns:
        tuple: A tuple containing the max and min IoU estimates."""
    max_common_intersection_sum = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sum",
    )

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Unpack improvement information
    improv_common_intersection, _, improv_common_extras, _, _, _ = max_improvement
    max_common_intersection = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    max_common_extras = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_extras",
        quantity_type="max",
        quantity_scope="sample",
    )

    # MaxIoU
    top_1_common_intersection = improv_common_intersection[0][TOP_INDEX_SAMPLE]
    top_1_common_extras = improv_common_extras[0][TOP_INDEX_SAMPLE]
    max_intersection = np.minimum(
        max_common_intersection, top_1_common_intersection
    ).sum()
    min_union = num_hits

    # Min IoU
    max_union = num_hits + np.minimum(max_common_extras, top_1_common_extras).sum()
    min_intersection = 0.0
    return (max_intersection, min_intersection), (max_union, min_union)


def and_not_chain_estimation(
    *,
    label_quantities,
    neuron_quantities,
    max_improvement,
    num_hits,
    max_size_mask=None,
):
    """Estimate the max and min IoU for an AND NOT chain based on the label quantities, neuron quantities and improvement information.
    Args:
        label_quantities (tuple): Quantities for the label.
        neuron_quantities (tuple): Quantities for the neuron.
        max_improvement (tuple): Information about the maximum improvement.
        num_hits (int): Number of hits for the current formula.
        max_size_mask (int): The maximum size of the mask. Not used in the sample heuristic but we keep it for consistency with the sum variant
    Returns:
        tuple: A tuple containing the max and min IoU estimates."""

    # Sum quantities
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

    # Samples quantities
    max_common_intersection = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="common_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )
    unique_intersection = optimal_utils.get_quantity(
        label_info=label_quantities,
        quantity_name="unique_intersection",
        quantity_type="max",
        quantity_scope="sample",
    )

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + unique_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Unpack improvement information
    _, _, _, _, improv_common_uncovered, _ = max_improvement
    _, neuron_common_tuple, _, _, _, _ = neuron_quantities
    _, neuron_common_sum = neuron_common_tuple

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
    top_1_uncovered_common = improv_common_uncovered[0][TOP_INDEX_SAMPLE]
    top_1_uncovered_common_sum = improv_common_uncovered[0][TOP_INDEX_SUM]
    if top_1_uncovered_common_sum < neuron_common_sum:
        max_intersection = (
            unique_intersection
            + np.minimum(max_common_intersection, top_1_uncovered_common)
        ).sum()
    else:
        max_intersection = unique_intersection_sum + max_common_intersection_sum
    min_union = num_hits + unique_extras_sum

    # Min IoU
    min_intersection = unique_intersection_sum

    max_union = num_hits + unique_extras_sum + max_common_extras_sum
    return (max_intersection, min_intersection), (max_union, min_union)
