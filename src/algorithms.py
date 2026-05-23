"""NetDissect implementation """

from collections import Counter

from src import metrics
from utils import general_utils

################# VANILLA NETDISSSECT ####################


def get_netdissect_scores(bitmaps, masks):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    netdissect_rank = {}
    candidate_concepts = range(len(masks))
    for concept in candidate_concepts:
        concept_mask = general_utils.parse_mask_by_type(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        netdissect_rank[concept] = concept_iou
    return netdissect_rank


def get_netdissect_explanation(bitmaps, masks):
    """Return the single best NetDissect concept explanation.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of samples.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).

    Returns:
        tuple: `(best_label, best_iou)` where `best_label` is the concept id
            with the highest IoU and `best_iou` is its score.
    """
    scores = get_netdissect_scores(bitmaps, masks)
    best_label = Counter(scores).most_common(1)[0][0]
    best_iou = Counter(scores).most_common(1)[0][1]
    return best_label, best_iou
