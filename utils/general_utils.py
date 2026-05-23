"""General-purpose utility functions used across the codebase.

Includes reproducibility, mask parsing, visualization, and helper routines.
"""

import random
import os

from numpy.random import RandomState
import scipy.sparse as sparse
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from utils import mask_utils


##### SEED UTILS #####
def set_seed(seed: int) -> RandomState:
    """Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # set to false for reproducibility, True to boost performance
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    return g


# reference: https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    """Method to set seed for each worker.
    Args:
        worker_id (int): Id of the worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#################### MASKS UTILS (Generic) ########################
def torch_to_sparse(vector):
    """
    Convert a torch tensor to a sparse matrix.

    Args:
        vector (torch.Tensor): tensor to convert

    Returns:
        scipy.sparse.csr_matrix: sparse matrix
    """
    if len(vector.shape) != 2:
        vector = torch.reshape(vector, (vector.shape[0], -1))
    return sparse.csr_matrix(vector.numpy())


def sparse_to_torch(vector):
    """
    Convert a sparse matrix to a torch tensor.

    Args:
        vector (scipy.sparse.csr_matrix): sparse matrix to convert

    Returns:
        torch.Tensor: tensor
    """
    return torch.from_numpy(vector.toarray())


def parse_mask_by_type(mask):
    """Parse the mask by type.
    - If the mask is a sparse matrix, convert it to a torch tensor.
    - If the mask is a string, load the sparse matrix from the path and convert it to a torch tensor.
    Otherwise, return the mask as is."""
    if isinstance(mask, sparse.csr.csr_matrix):
        return sparse_to_torch(mask)
    elif isinstance(mask, str):
        # this is a path string
        path = mask
        return sparse_to_torch(sparse.load_npz(path))
    else:
        return mask


############### VISUAL UTILS ####################


def get_grid_intersection(
    *, dataset, label, masks, bitmaps, number_samples, mask_shape, device
):
    """Get a grid of images with the intersection between the bitmaps and the label mask.

    Args:
        dataset (DatasetObject): dataset object including the imagess
        label (Formula): formula representing the label to compute the intersection with
        masks (list): list of masks representing the concepts
        bitmaps (torch.Tensor): tensor of shape (N, H*W) representing the bitmaps of the samples, where N is the number of samples, H and W are the height and width of the images, respectively.
        number_samples (int): number of samples to show in the grid
        mask_shape (tuple): tuple representing the shape of the masks, which is the same as the shape of the images (H, W)
        device (torch.device): device to use for the computations
    Returns:
        torch.Tensor: tensor of shape (3, H_grid, W_grid) representing the grid of images with the intersection between the bitmaps and the label mask, where H_grid and W_grid

    """

    label_mask = mask_utils.get_formula_mask(label, masks).to(device)

    # Compute the intersection between the bitmaps and the label mask
    # This will be used to compute the coverage of the bitmaps on the label mask, which will be used to select the samples to show in the grid.
    fire_and_label = bitmaps & label_mask

    # Sort the samples by coverage of the intersection between the bitmaps and the label mask
    fire_cov_per_sample = bitmaps.sum(dim=1) / bitmaps.shape[1]
    _, sorted_indices = torch.sort(fire_cov_per_sample, descending=True)

    # Extract the candidate indices
    if len(sorted_indices) > number_samples:
        selected_indices = sorted_indices[:number_samples]
    else:
        selected_indices = sorted_indices

    if len(selected_indices) > 0:
        segmented_images = []
        # For each selected sample, extract the image from the dataset and overlay the bitmaps and the label mask on top of it. The bitmaps are colored in yellow and the label mask is colored in blue.
        for index in selected_indices:

            # Load image from dataset and resize
            if isinstance(dataset[index], dict) and "file_name" in dataset[index]:
                path_image = dataset[index]["file_name"]
                image = torchvision.io.decode_image(
                    torchvision.io.read_file(path_image),
                    mode=torchvision.io.image.ImageReadMode.RGB,
                )
            else:
                image = torchvision.transforms.functional.pil_to_tensor(
                    dataset[index][0]
                )
            image = torchvision.transforms.functional.resize(image, mask_shape)

            # Overlay bitmaps and label mask on top of the image
            fire_image = fire_and_label[index].reshape(mask_shape[0], mask_shape[1])
            bitmaps_image = bitmaps[index].reshape(mask_shape[0], mask_shape[1])
            image_plus_bitmaps = torchvision.utils.draw_segmentation_masks(
                image, bitmaps_image, alpha=0.4, colors="yellow"
            )
            image_plus_mask = torchvision.utils.draw_segmentation_masks(
                image_plus_bitmaps, fire_image, alpha=0.6, colors="blue"
            )
            segmented_images.append(image_plus_mask)
        grid = torchvision.utils.make_grid(segmented_images, padding=2, pad_value=255)
    else:
        grid = None
    return grid


def get_figure(imgs, labels=None, width=8, height=4.8, hspace=0.5):
    """Get a figure with the images in a grid.
    Args:
        imgs (list): list of tensors of shape (3, H, W) representing the images to show in the grid, where H and W are the height and width of the images, respectively.
        labels (list): list of strings representing the labels to show on top of each image in the grid. If None, no labels will be shown.
        width (int): width of the figure
        height (int): height of the figure
        hspace (float): vertical space between the images in the grid
    Returns:
        matplotlib.figure.Figure: figure with the images in a grid"""

    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(labels, list):
        labels = [labels]
    fig, axs = plt.subplots(
        nrows=len(imgs), squeeze=False, gridspec_kw={"wspace": 0, "hspace": hspace}
    )
    fig.set_figwidth(width)
    fig.set_figheight(height)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if labels is not None:
            axs[i, 0].set_title(labels[i])
    return fig
