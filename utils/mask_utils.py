"""Mask loading, caching, and formula-mask composition utilities."""

import os
import pickle

import torch
from tqdm import tqdm
import scipy.sparse as sparse

from src import annotations_wrapper
from compositional import formula as F
from . import mmesh_utils
from . import general_utils


############### MASKS LOADING AND COMPUTING #######################
def load_masks_path(config):
    """
    Loads the sparse masks from the given directory.
    Args:
        config (src.config.Config): configuration of the current run.
    Returns:
        List of paths to the sparse masks and list of concept names.
    """
    dataset_name = config.get_dataset_name()
    masks_directory = config.get_annotations_dir()
    annotation_wrapper = annotations_wrapper.AnnotationsWrapper(
        dataset_name, config=config
    )
    concept_names = annotation_wrapper.concept_labels
    path_list = []
    for concept in tqdm(
        concept_names, desc="Loading Sparse Masks", total=len(concept_names)
    ):
        path_list.append(f"{masks_directory}/{concept}.npz")
    return path_list, concept_names


def get_masks(config):
    """Loads the sparse masks from the given directory.
    Args:
        config (src.config.Config): configuration of the current run.
    Returns:
        List of sparse masks and list of concept names.
    """
    # Useful variables
    dataset_name = config.get_dataset_name()

    # Compute or load masks and their labels
    annotation_wrapper = annotations_wrapper.AnnotationsWrapper(
        dataset_name, config=config
    )
    masks = annotation_wrapper.get_masks(config)
    labels = annotation_wrapper.concept_labels

    del annotation_wrapper

    return masks, labels


def get_concepts(config):
    """Loads the concept names from the given directory.
    Args:
        config (src.config.Config): configuration of the current run.
    Returns:
        List of concept names.
    """
    # Useful variables
    dataset_name = config.get_dataset_name()

    # Compute or load masks and their labels
    annotation_wrapper = annotations_wrapper.AnnotationsWrapper(
        dataset_name, config=config
    )
    labels = annotation_wrapper.concept_labels

    del annotation_wrapper

    return labels


def load_concept_mask(concept, segmentations_directory):
    """
    Loads the sparse mask of a concept from the given directory.
    Args:
        concept (str): name of the concept.
        segmentations_directory (str): directory where the masks are stored.
    Returns:
        Sparse mask of the concept.
    """

    return sparse.load_npz(f"{segmentations_directory}/{concept}.npz")


def load_sparse_masks(concept_names, segmentations_directory):
    """
    Loads the sparse masks from the given directory.
    Args:
        concept_names (list): list of concept names.
        segmentations_directory (str): directory where the masks are stored.
    Returns:
        List of sparse masks.
    """

    masks_list = []
    for concept in tqdm(
        concept_names, desc="Loading Sparse Masks", total=len(concept_names)
    ):
        masks_list.append(load_concept_mask(concept, segmentations_directory))
    return masks_list


############### MASKS INFO COMPUTING #######################


def get_disjoint_info(config):
    """Returns the disjoint information of the concepts.
    Args:
        config (src.config.Config): configuration of the current run.
    Returns:
        disjoint_info (torch.Tensor): tensor of shape (num_concepts, num_concepts) where disjoint_info[i,j] is True if concept i and concept j are disjoint, and False otherwise.
    """

    dataset_name = config.get_dataset_name()
    dir_info = config.get_info_dir()

    annotation_wrapper = annotations_wrapper.AnnotationsWrapper(
        dataset_name, config=config
    )
    disjoint_info = annotation_wrapper.compute_disjoint_info(dir_info)

    del annotation_wrapper

    return disjoint_info


def get_dataset_quantities(masks, info_directory):
    """Returns the quantities of the dataset, i.e. the common, unique and uncoverable elements of the masks.
    Args:
        masks (list): list of masks.
        info_directory (str): directory where to save/load the information.
    Returns:
        common_elements (torch.Tensor): tensor of shape (mask_shape) where common_elements[i] is True if the element i is covered by more than one mask, and False otherwise.
        unique_elements (torch.Tensor): tensor of shape (mask_shape) where unique_elements[i] is True if the element i is covered by exactly one mask, and False otherwise.
        uncoverable_elements (torch.Tensor): tensor of shape (mask_shape) where uncoverable_elements[i] is True if the element i is not covered by any mask, and False otherwise.
    """
    file_path = f"{info_directory}/quantities.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            common_elements, unique_elements, uncoverable_elements = pickle.load(file)
    else:
        # Compute unique elements
        sum_elements = torch.zeros_like(
            general_utils.parse_mask_by_type(masks[1]), dtype=torch.int32
        )
        for i in range(len(masks)):
            sum_elements += general_utils.parse_mask_by_type(masks[i])

        # In case of non overlapping annotations this should be equal to sum(sum_elements)
        common_elements = (sum_elements > 1).bool()
        unique_elements = (sum_elements == 1).bool()
        uncoverable_elements = (sum_elements == 0).bool()
        with open(file_path, "wb") as file:
            pickle.dump((common_elements, unique_elements, uncoverable_elements), file)
    return common_elements, unique_elements, uncoverable_elements


def get_masks_info(
    masks, config, bb_boxes=False, areas=False, inscribed=False, quantities=False
):
    """Returns the masks information useful for the heuristics.

    Args:
        masks (list): list of masks.
        config (src.config.Config): configuration of the current run.
        mask_shape (tuple): shape of the masks.
        device (torch.device): device to use to store the masks information.
        bb_boxes (bool): whether to compute the bounding boxes of the masks (required for mmesh, not useful for other methods).
        areas (bool): whether to compute the areas of the masks (required for mmesh, not useful for other methods).
        inscribed (bool): whether to compute the inscribed rectangles of the masks (required for mmesh, not useful for other methods).
        quantities (bool): whether to compute the quantities of the masks (common, unique, uncoverable). Required for optimal and beam optimal heuristics, not useful for mmesh.

    Returns:
        tuple: tuple containing:
            - concept_areas (list): list of areas of the masks.
            - (inscribed_rectangles, bounding_boxes) (tuple): tuple containing the inscribed rectangles and the bounding boxes of the masks.
            - quantities (tuple): tuple containing the common, unique and uncoverable elements of the masks
    """

    info_directory = config.get_info_dir()
    device = config.get_device()
    mask_shape = config.get_mask_shape()

    if not os.path.exists(info_directory):
        os.makedirs(info_directory)
    if areas:
        concept_areas = mmesh_utils.get_areas_mask(masks, info_directory, device)
    else:
        concept_areas = None
    if inscribed:
        inscribed_rectangles = mmesh_utils.get_concept_inscribed_masks(
            masks, mask_shape=mask_shape, info_directory=info_directory, device=device
        )
    else:
        inscribed_rectangles = None
    if bb_boxes:
        bounding_boxes = mmesh_utils.get_bounding_boxes(
            masks, mask_shape=mask_shape, info_directory=info_directory, device=device
        )
    else:
        bounding_boxes = None

    if quantities:
        quantities = get_dataset_quantities(masks, info_directory=info_directory)
    else:
        quantities = None

    masks_info = (concept_areas, (inscribed_rectangles, bounding_boxes), quantities)
    return masks_info


def get_formula_mask(f, masks, optional_masks=None):
    """
    Function to return a mask for a given formula.
    Args:
        f (src.formula.Formula): formula.
        masks (list): list of masks.
        optional_masks (dict): dictionary of additional masks (beam masks).
    Returns:
        Formula's Mask.
    """
    if optional_masks is not None and f in optional_masks.keys():
        mask = optional_masks[f]
        return general_utils.parse_mask_by_type(mask)
    if isinstance(f, F.Leaf):
        mask = masks[f.val]
        return general_utils.parse_mask_by_type(mask)
    elif isinstance(f, F.Or):
        masks_l = get_formula_mask(f.left, masks, optional_masks)
        masks_r = get_formula_mask(f.right, masks, optional_masks)
        return masks_l | masks_r
    elif isinstance(f, F.And):
        masks_l = get_formula_mask(f.left, masks, optional_masks)
        masks_r = get_formula_mask(f.right, masks, optional_masks)
        return masks_l & masks_r
    elif isinstance(f, F.Not):
        return ~get_formula_mask(f.val, masks, optional_masks)
    elif isinstance(f, int):
        mask = masks[f]
        return general_utils.parse_mask_by_type(mask)
    else:
        raise ValueError(f"Unknown formula type {type(f)}")


def get_formula_mask_and_tree(f, masks, path_masks=None):
    """
    Get the masks of the ancestors of a formula f including itself.
    Note that this is equivalent to get_formula_mask but it also returns the masks of the ancestors of f.
    Args:
        f (F.Formula): formula to get the ancestors masks
        masks (dict): dictionary of the masks of the leaf nodes
        path_masks (dict): dictionary of the masks of the ancestors already computed in the path from the root to the current node (optional)
    Returns:
        dict: dictionary of the masks of the ancestors of f including itself

    """
    if path_masks is not None and f in path_masks.keys():
        return path_masks
    elif path_masks is None:
        path_masks = {}
    if isinstance(f, F.Leaf):
        mask = masks[f.val]
        path_masks[f] = general_utils.parse_mask_by_type(mask)
        return path_masks
    elif isinstance(f, F.Or):
        l_ancestors_masks = get_formula_mask_and_tree(f.left, masks, path_masks)
        r_ancestors_masks = get_formula_mask_and_tree(f.right, masks, path_masks)
        mask = l_ancestors_masks[f.left] | r_ancestors_masks[f.right]
        path_masks[f] = mask
        path_masks.update(l_ancestors_masks)
        path_masks.update(r_ancestors_masks)
        return path_masks
    elif isinstance(f, F.And):
        l_ancestors_masks = get_formula_mask_and_tree(f.left, masks, path_masks)
        r_ancestors_masks = get_formula_mask_and_tree(f.right, masks, path_masks)
        mask = l_ancestors_masks[f.left] & r_ancestors_masks[f.right]
        path_masks[f] = mask
        path_masks.update(l_ancestors_masks)
        path_masks.update(r_ancestors_masks)
        return path_masks
    elif isinstance(f, F.Not):
        l_ancestors_masks = get_formula_mask_and_tree(f.val, masks, path_masks)
        not_mask = ~l_ancestors_masks[f.val]
        path_masks[f] = not_mask
        return path_masks
    elif isinstance(f, int):
        mask = masks[f]
        path_masks[F.Leaf(f)] = general_utils.sparse_to_torch(mask)
        return path_masks
    else:
        raise ValueError(f"Unknown formula type {type(f)}")
