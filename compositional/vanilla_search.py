"""Vanilla beam-search implementation without heuristics."""

from collections import Counter

from compositional import formula as F
from utils import search_utils
from src import algorithms


def compute_vanilla_explanations(*, masks, bitmaps, config):
    """Compute vanilla beam-search explanations.

    Args:
        masks (dict): A dictionary of concept masks indexed by concept id.
        bitmaps (torch.Tensor): A tensor of shape (N, H, W), where N is the
            number of samples.
        config (Settings): Configuration object providing beam size and formula length.

    Returns:
        tuple: `(best_label, best_iou, total_visited, expanded_nodes, estimated_nodes)`.
            If no positive-IoU seed concept exists, returns `(None, -1, -1, -1, -1)`.
    """
    # Get parameters from config
    beam_size = config.get_beam_limit()
    length = config.get_length()

    netdissect_rank = algorithms.get_netdissect_scores(bitmaps, masks)

    # Extract first beam and candidate concepts
    netdissect_rank = Counter(netdissect_rank)
    beam = {
        F.Leaf(lab): iou
        for lab, iou in netdissect_rank.most_common(beam_size * 2)
        if iou > 0
    }

    if len(beam) == 0:
        return None, -1, -1, -1, -1
    candidate_labels = [F.Leaf(lab) for lab in range(len(masks))]
    # Beam Search
    total_visited = 0
    expanded_nodes = 0
    estimated_nodes = 0
    beam_masks = {}

    for previous_beam_length in range(1, length):
        # Only expand formulas of the previous beam length to avoid regenerating beam tree already explored
        to_expland = [lab for lab in beam.keys() if len(lab) == previous_beam_length]
        next_state_space = search_utils.compute_next_search_space(
            to_expland, candidate_labels
        )

        # If we are in the last iteration, we set the beam size to 1 to get only the best formula
        if previous_beam_length == length - 1:
            beam_size = 1

        next_beam_formulas, next_beam_iou, beam_visited = search_utils.beam_search(
            next_state_space,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=beam_size,
        )

        expanded_nodes += len(to_expland)
        estimated_nodes += len(
            next_state_space
        )  # No heuristic, so we do not estimate the number of nodes to visit
        total_visited = total_visited + beam_visited

        # Update top formulas
        for index_beam in range(len(next_beam_formulas)):
            beam.update({next_beam_formulas[index_beam]: next_beam_iou[index_beam]})

        # Trim the beam
        beam = dict(Counter(beam).most_common(beam_size))
    top_result = Counter(beam).most_common(1)[0]

    best_iou = top_result[1]
    best_label = top_result[0]
    return best_label, best_iou, total_visited, expanded_nodes, estimated_nodes
