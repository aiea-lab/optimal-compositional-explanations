"""Generic beam-search helpers."""

import queue as Q

from compositional import formula as F
from utils import mask_utils
from src import metrics


def beam_search(
    search_space,
    *,
    masks,
    beam_masks,
    bitmaps,
    beam_limit,
    previous_beam=None,
):
    """Perform the beam search on the search space.

    Args:
        search_space (list): A list of formulas.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (N, H, W).
        beam_masks (dict): A dictionary of cached formula masks for formulas in
            the current beam.
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of samples.
        beam_limit (int): The beam size.
        previous_beam (dict): A dictionary mapping beam formulas to their IoU.

    Returns:
        current_beam_formulas (list): A list of formulas.
        current_beam_iou (list): A list of IoU values corresponding to
            `current_beam_formulas`.
        visited_indices (int): The number of formulas whose exact IoU was evaluated.
    """
    if previous_beam is None:
        previous_beam = {}
    current_beam = Q.PriorityQueue(beam_limit)
    current_beam_iou = []
    current_beam_formulas = []
    minimum = 0
    visited_indices = 0
    best_formula = None
    # Init beam with previous best
    for label, iou in previous_beam.items():
        if not current_beam.full():
            current_beam.put((iou, label))
            minimum = current_beam.queue[0][0]
        elif iou > minimum:
            current_beam.get()
            current_beam.put((iou, label))
            minimum = current_beam.queue[0][0]

    # Set minimum
    if current_beam.empty():
        minimum = 0
    else:
        minimum = current_beam.queue[0][0]

    # Iterate over the search space
    for candidate_formula in search_space:
        e_iou = candidate_formula.iou

        # If the estimated IoU is less than the minimum, we can stop the search if the beam is full
        if current_beam.full() and e_iou < minimum:
            break

        # skip equivalent formulas of the current beam
        if best_formula and hash(candidate_formula) == hash(best_formula):
            continue

        # Compute IoU
        masks_formula = mask_utils.get_formula_mask(
            candidate_formula, masks, beam_masks, device=bitmaps.device
        )
        iou = metrics.iou(masks_formula, bitmaps)

        # Update visited nodes
        visited_indices += 1

        # Update beam
        if not current_beam.full():
            candidate_formula.iou = iou
            current_beam.put((iou, candidate_formula))
            minimum = current_beam.queue[0][0]
        elif iou > minimum:
            candidate_formula.iou = iou
            current_beam.get()
            current_beam.put((iou, candidate_formula))
            minimum = current_beam.queue[0][0]

    # Extract formulas and iou from the beam
    for _ in range(current_beam.qsize()):
        iou, candidate = current_beam.get()
        current_beam_formulas.append(candidate)
        current_beam_iou.append(iou)
    return current_beam_formulas, current_beam_iou, visited_indices


def compute_next_search_space(formulas, candidate_labels):
    """Compute the next search space starting from the current beam
    of formulas.

    Args:
        formulas (list): A list of formulas.
        candidate_labels (list): A list of candidate labels.

    Returns:
        search_space (list): A list of formulas.
    """
    search_space = []

    for formula in formulas:
        vals_formula = set(formula.get_vals())
        for candidate_term in candidate_labels:
            # remove dummy cases with void masks or equivalent formulas
            if candidate_term.val in vals_formula:
                continue
            for op, negate in [(F.Or, False), (F.And, False), (F.And, True)]:
                candidate_to_attach = candidate_term
                if negate:
                    candidate_to_attach = F.Not(candidate_to_attach)
                candidate_formula = op(formula, candidate_to_attach)
                candidate_formula.iou = 1.0

                search_space.append(candidate_formula)
    search_space = list(set(search_space))
    return search_space
