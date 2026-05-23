"""Wrapper around dataset annotations for mask extraction and caching.

Provides disjointness computation and segmentation preprocessing helpers.
"""

import os
from tqdm import tqdm
import pickle

import torch
import numpy as np
import scipy.sparse as sparse

from utils import dataset_utils
from utils import mask_utils


def parse_concept(segm_data, concept_value, mask_shape):
    """Parses the concept from the segmentation data and returns the corresponding mask.
    Args:
        segm_data (torch.Tensor): segmentation data of shape (B, C, H, W) where B is the batch size, C is the number of annotations categories (e.g., scene, object, etc), H and W are the height and width of the segmentation masks.
        concept_value (int): value of the concept to be parsed. It is the index of the concept in the segmentation data.
        mask_shape (tuple): shape of the mask to be returned. It is a tuple of the form (H, W) where H and W are the height and width of the mask.
    Returns:
        concept_mask (torch.Tensor): mask of shape (B, H, W) where B is the batch size, H and W are the height and width of the mask. The mask is
        a binary tensor where the value is 1 if the concept is present in the corresponding pixel and 0 otherwise.
    """
    num_categories_segmentations = segm_data.shape[1]
    concept_mask = torch.zeros(
        (
            segm_data.shape[0],
            segm_data.shape[2],
            segm_data.shape[3],
        ),
        dtype=bool,
    ).cuda()
    for index_segmentation in range(num_categories_segmentations):
        # We unify the concept value across all the segmentation categories by checking if the concept value is present in any of the segmentation categories. This is because in some datasets, the same concept can be represented by different values in different segmentation categories. For example, in the broden dataset, the concept "car" can be represented by the value 1 in the "object" category and by the value 2 in the "scene" category. By unifying the concept value across all the segmentation categories
        concept_mask = concept_mask | (
            segm_data[:, index_segmentation] == concept_value
        )

    # We resize the concept mask to the desired shape using nearest neighbor interpolation,
    concept_mask = (
        torch.nn.functional.interpolate(
            concept_mask.float().unsqueeze(0),
            size=mask_shape,
            mode="nearest",
        )
        .bool()
        .squeeze(0)
    )
    return concept_mask


class AnnotationsWrapper:
    def __init__(self, dataset_name, config=None) -> None:
        if "broden" in dataset_name:
            self.data_wrapper = dataset_utils.BrodenWrapper(config)
        else:
            self.data_wrapper = dataset_utils.DetectronWrapper(dataset_name)

        self.concept_labels = self.data_wrapper.concept_labels

    def load_masks(self, segmentations_directory):
        """
        Loads the sparse masks from the given directory.
        Args:
            segmentations_directory (str): directory where the masks are stored.
        Returns:
            List of sparse masks.
        """
        return mask_utils.load_sparse_masks(
            self.concept_labels, segmentations_directory
        )

    def compute_disjoint_info(self, info_dir):
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)
        path_matrix = f"{info_dir}/disjoint_matrix.pt"
        if os.path.exists(path_matrix):
            disjoint_matrix = pickle.load(open(path_matrix, "rb"))
            return disjoint_matrix

        print("Disjoint matrix not computed, computing it")
        # Collect segmentation masks for the whole dataset
        dataset_segmentations = []
        for data in tqdm(self.data_wrapper.data_loader, desc="Computing Segmentations"):
            segmentations = self.data_wrapper.extract_segmentations(data)
            dataset_segmentations.append(segmentations)

        # Create a dictionary to store the disjoint information
        disjoint_matrix = np.ones(
            (len(self.concept_labels), len(self.concept_labels)), dtype=bool
        )
        for concept in range(len(self.concept_labels)):
            disjoint_matrix[concept, concept] = False

        for segmentations in tqdm(
            dataset_segmentations, desc="Computing Disjoint Info"
        ):
            multiple_segmentations = (segmentations > 0).sum(axis=(1)) > 1
            index_multiple_segmentations = multiple_segmentations.nonzero()
            for b, h, w in index_multiple_segmentations:
                overlapping_concepts = segmentations[b, :, h, w].unique()
                # Remove the concepts overlapping from the disjoint_dict
                for index_over, concept_in_overlap in enumerate(overlapping_concepts):
                    concept_in_overlap = concept_in_overlap.item()
                    for other_concept in overlapping_concepts[index_over + 1 :]:
                        other_concept = other_concept.item()
                        disjoint_matrix[concept_in_overlap, other_concept] = False
                        disjoint_matrix[other_concept, concept_in_overlap] = False

        with open(path_matrix, "wb") as f:
            pickle.dump(disjoint_matrix, f)
        return disjoint_matrix

    def fast_extractions(self, concepts, mask_shape, dataset_annotations):
        masks = [[] for _ in range(len(self.concept_labels))]
        for concept_index in concepts:
            concept_mask = []
            for segmentations in dataset_annotations:
                segmentations = segmentations.cuda()
                concept_mask.append(
                    parse_concept(segmentations, concept_index, mask_shape)
                )
            concept_mask = torch.cat(concept_mask, 0)
            masks[concept_index].append(concept_mask.cpu())
        return masks

    def slow_extractions(self, concepts, mask_shape):
        masks = [[] for _ in range(len(self.concept_labels))]
        for data in self.data_wrapper.data_loader:
            # Extract the annotations from the data
            segmentations = self.data_wrapper.extract_segmentations(data)
            segmentations = segmentations.cuda()

            # Compute the masks for the selected concepts
            for concept_index in concepts:
                concept_mask = parse_concept(segmentations, concept_index, mask_shape)
                masks[concept_index].append(concept_mask.cpu())
        return masks

    def save_segmentation_masks(
        self, segmentation_dir, mask_shape, step_size=20, missing=None, fast_impl=True
    ):
        """
        Saves the segmentation masks for the given dataloader and labels in the given directory.
        Args:
            segmentation_dir (str): directory where to save the masks.
            mask_shape (tuple): shape of the masks to be saved.
            step_size (int): step size to be used for computing the masks. It is used to compute the masks in batches of concepts, to reduce the memory usage. It is ignored if the masks are already computed and stored in the directory.
            missing (list): list of concept indices for which the masks are missing and need to be computed. If None, it computes the masks for all
                the concepts. It is used to compute only the missing masks if some masks are already computed and stored in the directory, to allow resuming the computation if it was interrupted.
            fast_impl (bool): whether to use the fast implementation to compute the masks. The fast implementation requires more RAM but it is faster. If you have memory issues, set it to False.
        Returns:
                None
        """
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)
        if fast_impl:
            # Collect segmentation masks for the whole dataset
            dataset_segmentations = []
            for data in tqdm(self.data_loader, desc="Computing Segmentations"):
                segmentations = self.data_wrapper.concept_labels(data)
                dataset_segmentations.append(segmentations)
        else:
            dataset_segmentations = None

        tot_concepts = len(self.concept_labels)
        ranges = range(0, tot_concepts, step_size)
        for starting_index in tqdm(ranges):
            concepts = range(
                starting_index, min(starting_index + step_size, tot_concepts)
            )

            # Remove missing from concepts
            concepts = [concept for concept in concepts if concept in missing]

            # Compute the masks for the selected concepts
            if len(concepts) > 0:
                if fast_impl:
                    masks = self.fast_extractions(
                        concepts, mask_shape, dataset_segmentations
                    )
                else:
                    masks = self.slow_extractions(concepts, mask_shape)

                # Save the masks for the selected concepts
                for concept_index in sorted(range(len(masks))):
                    if len(masks[concept_index]) > 0 and concept_index < len(
                        self.concept_labels
                    ):
                        # Prepare for scipy sparse matrix format
                        masks[concept_index] = torch.cat(masks[concept_index], 0)
                        masks[concept_index] = torch.reshape(
                            masks[concept_index], (masks[concept_index].shape[0], -1)
                        )
                        masks[concept_index] = masks[concept_index].numpy()

                        with open(
                            f"{segmentation_dir}/"
                            + f"{self.concept_labels[concept_index]}.npz",
                            "wb",
                        ) as file:
                            sparse.save_npz(
                                file, sparse.csr_matrix(masks[concept_index])
                            )
                        masks[concept_index] = (
                            None  # Save memory by deleting the masks that are already saved
                        )
            del masks  # Save memory by deleting the masks of the current batch before moving to the next batch

    def get_masks(self, config):
        """
        Returns the masks for the given dataloader and labels.
        Args:
            config (dict): configuration dictionary containing the following keys:
                - masks_directory (str): directory where the sparse masks are stored.
                - mask_shape (tuple): shape of the masks to be returned.
                - dataset_name (str): name of the dataset for which to return the masks.
                - step_size (int): step size to be used for computing the masks. It is used to compute the masks in batches of concepts, to reduce the memory usage. It is ignored if the masks are already computed and stored in the directory.
        Returns:
            List of masks.
        """

        # Extract info
        masks_directory = config.get_annotations_dir()
        mask_shape = config.get_mask_shape()
        dataset_name = config.get_dataset_name()
        step_size = config.get_step_size()
        if step_size is None:
            step_size = len(self.concept_labels)
        else:
            step_size = min(step_size, len(self.concept_labels))

        # Extract ignore list
        if "broden" in dataset_name:
            ignore = [""]
        else:
            ignore = []

        # If some file is missing, generate them. This allows to resume the generation of the masks if it was interrupted, without having to recompute all the masks.
        missing = []
        for index, concept in enumerate(self.concept_labels):
            if os.path.exists(f"{masks_directory}/{concept}.npz"):
                continue
            else:
                missing.append(index)
        if len(missing) > 0:
            # Generate the missing masks
            print(f"Missing {len(missing)} masks in {masks_directory}")
            self.save_segmentation_masks(
                masks_directory, mask_shape, step_size, missing=missing, fast_impl=False
            )

        # Load the masks
        masks = self.load_masks(masks_directory)
        for i in range(len(masks)):
            # Zero out the masks that need to be ignored
            if ignore is not None and self.concept_labels[i] in ignore:
                masks[i] = sparse.csr_matrix(
                    torch.zeros_like(torch.from_numpy(masks[i].toarray())).numpy()
                )

        return masks
