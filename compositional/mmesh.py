"""Implementation of the M-MESH compositional search baseline."""

from collections import Counter

import torch

from src import metrics
from compositional import formula as F
from utils import general_utils, search_utils, mmesh_utils, mask_utils


def get_augmented_netdissect_scores(bitmaps, masks, *, device=torch.device("cpu")):
    """Compute the NetDissect score for each concept in the masks and additional info.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the
            number of sample. The bitmaps should be resized to the same size of the masks before passing them to the function.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H*W).
        device (torch.device): The device to use.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
        areas (list): A list of intersection areas.
    """
    netdissect_rank = {}
    areas = []
    candidate_concepts = range(len(masks))
    for concept in candidate_concepts:
        concept_mask = general_utils.parse_mask_by_type(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        intersection_area = (
            (concept_mask & bitmaps).sum(dim=1, dtype=torch.int32).to(device)
        )
        netdissect_rank[concept] = concept_iou
        areas.append(intersection_area)

    netdissect_rank = Counter(netdissect_rank)
    return netdissect_rank, areas


def formula_is_in(f, unary_areas, max_mask_size, enneary_areas=None):
    """
    Function to check where the formula is present in the data.

    Args:
        unary_areas (list): list of areas of the masks.
        f (src.formula.Formula): formula to check.
        max_mask_size (int): maximum size of the mask.
        enneary_areas (dict): dictionary of additional areas.

    Returns:
        Boolean list
    """
    if enneary_areas is not None and f in enneary_areas.keys():
        return enneary_areas[f] > 0
    if isinstance(f, F.And):
        masks_l = formula_is_in(f.left, unary_areas, max_mask_size, enneary_areas)
        masks_r = formula_is_in(f.right, unary_areas, max_mask_size, enneary_areas)
        return masks_l & masks_r
    elif isinstance(f, F.Or):
        masks_l = formula_is_in(f.left, unary_areas, max_mask_size, enneary_areas)
        masks_r = formula_is_in(f.right, unary_areas, max_mask_size, enneary_areas)
        return masks_l | masks_r
    elif isinstance(f, F.Not):
        return unary_areas[f.val.val] < max_mask_size
    elif isinstance(f, F.Leaf):
        return unary_areas[f.val] > 0


def get_coordinates(term, leaf_list, enneary_list):
    """Returns the coordinates of  of the formula
    Args:
        term (src.formula.Formula): term to check.
        leaf_list (dict): dictionary of leaf coordinates.
        enneary_list (dict): dictionary of enneary formulas coordinates.

    Returns:
        Coordinates of the formula
    """
    if isinstance(term, F.BinaryNode):
        coordinates = enneary_list[term]
    elif isinstance(term, F.Leaf):
        coordinates = leaf_list[term.val]
    else:
        coordinates = term
    return coordinates


def get_intersection_info(term, unary_intersect, enneary_intersect, neuron_areas):
    """
    Returns the intersection between the term and the firing areas
    Args:
        term (src.formula.Formula): term to check.
        unary_intersect (dict): dictionary of unary intersections.
        enneary_intersect (dict): dictionary of enneary intersections.
        neuron_areas (torch.tensor): tensor of neuron areas.

    Returns:
        Intersection between the term and the firing areas
    """

    if isinstance(term, F.BinaryNode):
        term_and_fires_areas = enneary_intersect[term]
    elif isinstance(term, F.Not):
        term_and_fires_areas = neuron_areas - unary_intersect[term.val.val]
    else:
        term_and_fires_areas = unary_intersect[term.val]

    return term_and_fires_areas


def get_area_info(term, unary_areas, enneary_areas, max_size_mask):
    """
    Returns the area of the term
    Args:
        term (src.formula.Formula): term to check.
        unary_areas (dict): dictionary of unary areas.
        enneary_areas (dict): dictionary of enneary areas.
        max_size_mask (int): maximum size of the mask.

    Returns:
        Area of the term
    """
    if isinstance(term, F.BinaryNode):
        areas = enneary_areas[term]
    elif isinstance(term, F.Not):
        areas = max_size_mask - get_area_info(
            term.val, unary_areas, enneary_areas, max_size_mask
        )
    else:
        areas = unary_areas[term.val]
    return areas


def is_scene(areas, max_size_mask):
    """Returns True if the mask is a scene, False otherwise"""
    condition = (areas == 0) | (areas == max_size_mask)
    if condition.all():
        flag = True
    else:
        flag = False
    return flag


def compute_scene_iou(
    formula,
    left_areas,
    right_areas,
    left_intersection_area,
    right_intersection_area,
    neuron_areas,
    max_size_mask,
    num_hits,
):
    """Computes the IoU of a scene formula
    Args:
        formula (src.formula.Formula): formula to check.
        left_areas (torch.tensor): sample areas of the left term.
        right_areas (torch.tensor): sample areas of the right term.
        left_intersection_area (torch.tensor): intersection areas
            with the neuron of the left term.
        right_intersection_area (torch.tensor): intersection areas
            with the neuron of the right term.
        neuron_areas (torch.tensor): tensor of neuron areas.
        max_size_mask (int): maximum size of the mask.
        num_hits (int): number of hits.
    Returns:
        IoU of the scene formula
    """
    if isinstance(formula, F.Or):
        # exact computation
        formula_mask = torch.where(left_areas > right_areas, left_areas, right_areas)
        intersection = torch.where(
            formula_mask == max_size_mask,
            neuron_areas,
            left_intersection_area + right_intersection_area,
        )

        intersection = torch.sum(intersection)
    elif isinstance(formula, F.And):
        formula_mask = torch.minimum(left_areas, right_areas)
        intersection = torch.where(
            left_intersection_area < right_intersection_area,
            left_intersection_area,
            right_intersection_area,
        )
        intersection = torch.sum(intersection)
    estimated_iou = intersection / (num_hits + torch.sum(formula_mask) - intersection)
    return torch.round(estimated_iou, decimals=4)


def mmesh_heuristic(formula, heuristic_info, *, num_hits, max_size_mask):
    """
    Computes the IoU of a formula using the mmesh heuristic.
    Args:
        formula (src.formula.Formula): formula to check.
        heuristic_info (tuple): tuple of unary and enneary heuristic_info collected from
            the dataset and the parsing of the previous beam.
        num_hits (int): number of hits in the neuron's mask.
        max_size_mask (int): maximum size of the mask.
    Returns:
        float: estimated iou
    """
    dissect_info = heuristic_info[0]
    enneary_info = heuristic_info[1]
    unary_info, neuron_areas, unary_intersection = dissect_info
    unary_areas, (unary_inscribed, unary_bounding_box) = unary_info
    enneary_areas, enneary_inscribed, enneary_bounding_box, enneary_intersection = (
        enneary_info
    )

    formula_in = formula_is_in(formula, unary_areas, max_size_mask, enneary_areas)

    left_and_fires_areas = get_intersection_info(
        formula.left, unary_intersection, enneary_intersection, neuron_areas
    )
    right_and_fires_areas = get_intersection_info(
        formula.right, unary_intersection, enneary_intersection, neuron_areas
    )
    left_areas = get_area_info(formula.left, unary_areas, enneary_areas, max_size_mask)
    right_areas = get_area_info(
        formula.right, unary_areas, enneary_areas, max_size_mask
    )

    left_intersection_area = left_and_fires_areas * formula_in
    right_intersection_area = right_and_fires_areas * formula_in

    # In case of scene formula, we can compute the exact formula mask
    # and in the OR case, we can compute the exact intersection
    left_is_scene = is_scene(left_areas, max_size_mask)
    right_is_scene = is_scene(right_areas, max_size_mask)
    if left_is_scene or right_is_scene:
        return compute_scene_iou(
            formula,
            left_areas,
            right_areas,
            left_intersection_area,
            right_intersection_area,
            neuron_areas,
            max_size_mask,
            num_hits,
        )

    # Otherswise, we have to approximate both of them
    if isinstance(formula, F.Or):
        max_intersection_neuron = torch.minimum(
            neuron_areas, left_intersection_area + right_intersection_area
        )
        minimum_area_mask = torch.maximum(left_areas, right_areas)
        coordinates_left = get_coordinates(
            formula.left, unary_bounding_box, enneary_bounding_box
        )
        coordinates_right = get_coordinates(
            formula.right, unary_bounding_box, enneary_bounding_box
        )
        maximum_intersection = mmesh_utils.get_rectangles_overlap(
            coordinates_left, coordinates_right
        )
        minimum_area_mask = torch.maximum(
            minimum_area_mask,
            left_areas + right_areas - maximum_intersection,
        )
        minimum_area_mask = torch.maximum(minimum_area_mask, max_intersection_neuron)
    elif isinstance(formula, F.And):
        max_intersection_neuron = torch.minimum(
            left_intersection_area, right_intersection_area
        )
        if isinstance(formula.right, F.Not):
            coordinates_left = get_coordinates(
                formula.left, unary_bounding_box, enneary_bounding_box
            )
            coordinates_right = get_coordinates(
                formula.right.val, unary_bounding_box, enneary_bounding_box
            )
            maximum_intersection = mmesh_utils.get_rectangles_overlap(
                coordinates_left, coordinates_right
            )
            minimum_area_mask = left_areas - maximum_intersection

        else:
            coordinates_left = get_coordinates(
                formula.left, unary_inscribed, enneary_inscribed
            )
            coordinates_right = get_coordinates(
                formula.right, unary_inscribed, enneary_inscribed
            )
            minimum_area_mask = mmesh_utils.get_rectangles_overlap(
                coordinates_left, coordinates_right
            )
        minimum_area_mask = torch.maximum(minimum_area_mask, max_intersection_neuron)

    max_intersection_neuron = torch.sum(max_intersection_neuron)
    minimum_area_mask = torch.sum(minimum_area_mask)
    estimated_iou = max_intersection_neuron / (
        num_hits + minimum_area_mask - max_intersection_neuron
    )
    return torch.round(estimated_iou, decimals=4)


def get_beam_info(beam, masks, bitmaps, mask_shape, device):
    """Compute the heuristic info for the beam.

    Args:
        heuristic (str): The heuristic to use.
        beam (dict): A dictionary of formulas of the current beam.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (N, H, W).
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        mask_shape (tuple): The shape of the mask.
        device (torch.device): The device to use.

    Returns:
        beam_masks (dict): A dictionary of labal masks of the formulas in the
        current beam. Each mask is a tensor of shape (N, H, W).
        updated_info (dict): A dictionary of heuristic info.
    """

    beam_masks = {}
    areas = {}
    intersections = {}
    inscribed = {}
    rectangles = {}

    # update infos
    for formula in beam:
        if isinstance(formula, F.Leaf):
            # Skip leaf. We have already them in masks
            continue

        # Compute formula mask
        masks_formula = mask_utils.get_formula_mask(formula, masks)
        mask_type = "tensor" if isinstance(masks_formula, torch.Tensor) else "sparse"
        if mask_type == "torch":
            beam_masks[formula] = masks_formula.to(device)
        elif mask_type == "sparse":
            beam_masks[formula] = general_utils.torch_to_sparse(masks_formula).to(
                device
            )
        masks_formula = masks_formula.to(bitmaps.device)
        # Compute heuristic info
        areas[formula] = masks_formula.sum(1, dtype=torch.int32).to(device)
        intersections[formula] = (
            (masks_formula & bitmaps).sum(1, dtype=torch.int32).to(device)
        )
        masks_formula = masks_formula.reshape(-1, mask_shape[0], mask_shape[1])
        inscribed[formula] = mmesh_utils.get_inscribed_rectangles(
            masks_formula, device
        ).to(device)
        rectangles[formula] = mmesh_utils.get_overscribed_rectangles(
            masks_formula, mask_shape
        ).to(device)
    return beam_masks, (areas, inscribed, rectangles, intersections)


def sort_search_space_by(search_space, *, heuristic_info, num_hits, max_size_mask):
    """
    Sort the search space using the heuristic name_heuristic.

    Args:
        search_space (list of Formula): the search space to sort
        heuristic_info (tuple): the information to be used by the heuristic
        num_hits (int): the number of hits of the neuron
        max_size_mask (int): the maximum size of the mask

    Returns:
        list of Formula: the sorted search space
    """
    for index_formula, candidate_formula in enumerate(search_space):
        # Estimate IoU
        esti = mmesh_heuristic(
            candidate_formula,
            heuristic_info,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
        )
        search_space[index_formula].iou = esti
    search_space = sorted(search_space, key=lambda x: x.iou, reverse=True)
    return search_space


def perform_search(
    *,
    netdissect_rank,
    masks,
    bitmaps,
    heuristic_info,
    max_size_mask,
    beam_size=5,
    length=3,
    mask_shape=None,
    device=torch.device("cpu"),
):
    """Compute the best formula using a beam search with the mmesh heuristic.

    Args:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H*W).
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the
            number of sample.
        heuristic_info (tuple): the information to be used by the heuristic
        max_size_mask (int): the maximum size of the mask
        beam_size (int): The beam size for the search.
        length (int): The length of the search.
        mask_shape (tuple): The shape of the masks.
        device (torch.device): The device to use for the computation.
    """

    # Extract first beam and candidate concepts
    beam = {
        F.Leaf(lab): iou
        for lab, iou in netdissect_rank.most_common(beam_size * 2)
        if iou > 0
    }
    if len(beam) == 0:
        return None, -1, -1, -1, -1

    # Number of hits
    num_hits = bitmaps.sum()

    # Beam Search
    candidate_labels = [F.Leaf(lab) for lab in range(len(masks))]
    total_visited = 0
    expanded_nodes = 0
    estimated_nodes = 0
    beam_masks = {}
    updated_info = (None, None, None, None)
    for previous_beam_length in range(1, length):

        # Only expand formulas of the previous beam length to avoid regenerating beam tree already explored
        to_expand = [lab for lab in beam.keys() if len(lab) == previous_beam_length]

        # Expand beam and sort search space
        sorted_search_space = sort_search_space_by(
            search_utils.compute_next_search_space(to_expand, candidate_labels),
            heuristic_info=(heuristic_info, updated_info),
            num_hits=num_hits,
            max_size_mask=max_size_mask,
        )

        # If we are in the last iteration, we set the beam size to 1 to get only the best formula
        if previous_beam_length == length - 1:
            beam_size = 1

        # Perform beam search
        next_beam_formulas, next_beam_iou, beam_visited = search_utils.beam_search(
            sorted_search_space,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=beam_size,
        )
        # Update statistics
        expanded_nodes += len(beam.keys())
        estimated_nodes += len(sorted_search_space)
        total_visited = total_visited + beam_visited

        # Update top formulas
        for index_beam in range(len(next_beam_formulas)):
            beam.update({next_beam_formulas[index_beam]: next_beam_iou[index_beam]})

        # Trim the beam
        beam = dict(Counter(beam).most_common(beam_size))

        # Update infos if there are step left
        if previous_beam_length < length - 1:
            beam_masks, updated_info = get_beam_info(
                beam, masks, bitmaps, mask_shape, device
            )
    top_result = Counter(beam).most_common(1)[0]

    best_iou = top_result[1]
    best_label = top_result[0]
    return best_label, best_iou, total_visited, expanded_nodes, estimated_nodes


def compute_mmesh_explanations(*, bitmaps, masks, masks_info, config):
    # Get parameters from config
    beam_size = config.get_beam_limit()
    length = config.get_length()
    max_size_mask = config.get_mask_shape()[0] * config.get_mask_shape()[1]
    device = config.get_device()
    mask_shape = config.get_mask_shape()

    # Compute IoU leaf concepts and additional info
    sample_activation_areas = bitmaps.sum(1).to(device)
    netdissect_scores, intersect_areas = get_augmented_netdissect_scores(
        bitmaps, masks, device=device
    )
    heuristic_info = (
        (masks_info[0], (masks_info[1][0], masks_info[1][1])),
        sample_activation_areas,
        intersect_areas,
    )

    # Run Search
    best_label, best_iou, visited, expanded, estimated = perform_search(
        netdissect_rank=netdissect_scores,
        masks=masks,
        bitmaps=bitmaps,
        heuristic_info=heuristic_info,
        beam_size=beam_size,
        length=length,
        max_size_mask=max_size_mask,
        mask_shape=mask_shape,
        device=device,
    )
    return best_label, best_iou, visited, expanded, estimated
