"""Beam-search variant guided by the optimal heuristic family."""

from collections import Counter

from compositional import formula as F
from compositional import optimal_sample_heuristic
from utils import optimal_utils, search_utils, mask_utils


def compute_beam_quantities(beam, masks, beam_masks, heuristic_info, bitmaps):
    """Computes the quantities for each formula in the beam.
    Args:
        beam (dict): A dictionary containing the formulas in the beam and their corresponding iou.
        masks (list): A list of concept masks. Each mask is a tensor of shape (H*W).
        beam_masks (dict): A dictionary containing the masks for each formula in the beam.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
    Returns:
        tuple: A tuple containing the beam info and the updated beam masks.
    """
    seg_quantities, neuron_quantities, _ = heuristic_info
    (neuron_unique, _), (neuron_common, _), _, _, _, _ = neuron_quantities
    common_elements, unique_elements, _ = seg_quantities

    # Elements re-used by all the nodes, we pre-move them to gpu if available
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    beam_info = {}
    for label in beam:
        if label in beam_masks or isinstance(label, F.Leaf):
            # We already have the information for this label, we skip it
            continue

        # Compute the mask for the label
        label_mask = mask_utils.get_formula_mask(label, masks, beam_masks).to(
            bitmaps.device
        )

        # Compute the quantities from the mask
        label_quantities = optimal_utils.compute_quantities_vector(
            label_mask=label_mask,
            bitmaps=bitmaps,
            common_elements=common_elements,
            unique_elements=unique_elements,
            neuron_common=neuron_common,
            neuron_unique=neuron_unique,
        )

        # Compute the info from the quantities
        label_info = optimal_utils.get_concept_info(label_quantities)

        beam_info[label] = label_info
        label_mask = label_mask.cpu()
        beam_masks[label] = label_mask
    return beam_info, beam_masks


def get_beam_info(
    *, beam, masks, beam_masks, heuristic_info, label_mapping, bitmaps, length
):
    """Gets the information for each formula in the beam and updates the heuristic information and the label mapping.
    Args:
        beam (dict): A dictionary containing the formulas in the beam and their corresponding iou.
        masks (list): A list of concept masks. Each mask is a tensor of shape (H*W).
        beam_masks (dict): A dictionary containing the masks for each formula in the beam.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        label_mapping (dict): A dictionary mapping labels to the indices of their corresponding masks.
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        length (int): The maximum length of the formulas to search.
    Returns:
        tuple: A tuple containing the updated beam masks and the updated heuristic information and label mapping.
    """
    beam_info, beam_masks = compute_beam_quantities(
        beam, masks, beam_masks, heuristic_info, bitmaps
    )
    new_heuristic_info, new_label_mapping = optimal_utils.update_heuristic_info(
        nodes_info=beam_info,
        heuristic_info=heuristic_info,
        label_mapping=label_mapping,
        max_length=length,
    )
    return beam_masks, (new_heuristic_info, new_label_mapping)


def sort_search_space_by(
    *,
    search_space,
    label_mapping,
    heuristic_info,
    disjoint_info,
    num_hits,
    max_size_mask
):
    """Sorts the search space based on the estimated iou for each formula in the search space.
    Args:
        search_space (list): A list of formulas to sort.
        label_mapping (dict): A dictionary mapping labels to the indices of their corresponding masks.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        num_hits (int): The number of hits in the bitmaps.
        max_size_mask (int): The maximum size of the mask.
    Returns:
        list: A sorted list of formulas based on the estimated iou.
    """
    _, neuron_quantities, _ = heuristic_info
    for index_formula, candidate_formula in enumerate(search_space):
        label_quantities = optimal_utils.estimate_label_quantities(
            heuristic=optimal_sample_heuristic,
            label=candidate_formula,
            label_mapping=label_mapping,
            heuristic_info=heuristic_info,
            max_size_mask=max_size_mask,
            disjoint_info=disjoint_info,
        )
        if label_quantities is None:
            # Label discarded at the previous step
            esti_iou = 0.0
        else:

            esti_iou = optimal_utils.compute_max_iou_from_label_info(
                label_quantities=label_quantities,
                num_hits=num_hits,
                neuron_quantities=neuron_quantities,
            )
        search_space[index_formula].iou = esti_iou

    # Sort the search space based on the estimated iou in descending order
    search_space = sorted(search_space, key=lambda x: x.iou, reverse=True)
    return search_space


def perform_search(
    *,
    masks,
    bitmaps,
    heuristic_info,
    disjoint_info,
    max_size_mask,
    beam_size=5,
    length=3
):
    """Performs the beam optimal search to find the best formula that explains the bitmaps given the masks and the heuristic information.
    Args:
        masks (list): A list of concept masks. Each mask is a tensor of shape (H*W).
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        heuristic_info (tuple): A tuple containing the masks_info, neuron_quantities, and concept quantities for the optimal heuristic.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        max_size_mask (int): The maximum size of the mask.
        beam_size (int): The beam size for the search.
        length (int): The maximum length of the formulas to search.
    Returns:
        tuple: A tuple containing the best formula, its iou, the total number of visited nodes, the number of expanded nodes, and the number of estimated nodes.
    """

    # Number of hits
    num_hits = bitmaps.sum()

    # Utilities
    label_mapping = {}
    leaf_mapping = {F.Leaf(c): c for c in range(len(masks))}
    label_mapping.update(leaf_mapping)

    # Extract first beam and candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    _, neuron_quantities, concept_quantities = heuristic_info
    iou_atoms = {
        k: optimal_utils.compute_max_iou_from_label_info(
            concept_quantities[label_mapping[k]], neuron_quantities, num_hits
        )
        for k in candidate_labels
    }
    iou_atoms = Counter(iou_atoms)
    first_beam_num = min(len(iou_atoms), beam_size * 2)
    beam_atoms = {
        lab: iou for lab, iou in iou_atoms.most_common(first_beam_num) if iou > 0
    }

    if len(beam_atoms) == 0:
        return None, -1, -1, -1, -1

    # Beam Search
    total_visited = 0
    expanded_nodes = 0
    estimated_nodes = 0
    beam_masks = {}
    beam = beam_atoms.copy()
    for previous_beam_length in range(1, length):
        # Only expand formulas of the previous beam length to avoid regenerating beam tree already explored
        to_expand = [lab for lab in beam.keys() if len(lab) == previous_beam_length]
        search_space = search_utils.compute_next_search_space(
            to_expand,
            candidate_labels,
        )

        # Compute the estimation for the next frontier
        sorted_search_space = sort_search_space_by(
            search_space=search_space,
            label_mapping=label_mapping,
            heuristic_info=heuristic_info,
            num_hits=num_hits,
            max_size_mask=max_size_mask,
            disjoint_info=disjoint_info,
        )

        # # If we are in the last iteration, we set the beam size to 1 to get only the best formula
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
        expanded_nodes += len(to_expand)
        estimated_nodes += len(sorted_search_space)
        total_visited = total_visited + beam_visited

        # Update top formulas
        for index_beam in range(len(next_beam_formulas)):
            beam.update({next_beam_formulas[index_beam]: next_beam_iou[index_beam]})

        # Trim the beam
        beam = dict(Counter(beam).most_common(beam_size))

        # Update infos if there are step left
        if previous_beam_length < length - 1:
            beam_masks, (heuristic_info, label_mapping) = get_beam_info(
                beam=beam.keys(),
                masks=masks,
                beam_masks=beam_masks,
                heuristic_info=heuristic_info,
                label_mapping=label_mapping,
                bitmaps=bitmaps,
                length=length,
            )

    top_result = Counter(beam).most_common(1)[0]

    best_iou = top_result[1]
    best_label = top_result[0]
    return best_label, best_iou, total_visited, expanded_nodes, estimated_nodes


def compute_beam_optimal_explanations(
    *, bitmaps, masks, masks_info, disjoint_info, config
):
    """Computes the beam optimal explanations for the given bitmaps and masks.
    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        masks (list): A list of concept masks. Each mask is a tensor of shape (H*W).
        masks_info (list): A list of dictionaries containing the information about the masks.
        disjoint_info (dict): A dictionary containing the disjoint information for the concepts.
        config (Config): A configuration object containing the parameters for the search.
    Returns:
        tuple: A tuple containing the best formula, its iou, the total number of visited nodes, the number of expanded nodes, and the number of estimated nodes.
    """
    # Extract info from config
    beam_size = config.get_beam_limit()
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
        beam_size=beam_size,
        length=length,
    )
    return best_label, best_iou, visited, expanded, estimated
