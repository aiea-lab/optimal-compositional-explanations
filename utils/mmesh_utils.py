# Reference: https://github.com/KRLGroup/Clustered-Compositional-Explanations/tree/main
"""Geometry and mask helpers used by the M-MESH implementation."""

import os
import pickle

import torch
import torchvision
from tqdm import tqdm

from . import general_utils


def get_areas_mask(masks, info_directory, device=torch.device("cpu")):
    """
    Returns the areas per sample of the masks for each atomic concept.
    Args:
        masks (list): list of masks.
        info_directory (str): directory where the information is stored.
        device (torch.device): device to use.
    Returns:
        List of areas of the masks.
    """
    areas = []
    file_concept_areas = f"{info_directory}/concept_areas_list.pkl"
    if os.path.exists(file_concept_areas):
        with open(file_concept_areas, "rb") as file:
            areas = pickle.load(file)
    else:
        for concept in range(len(masks)):
            areas.append(
                torch.sum(
                    general_utils.parse_mask_by_type(masks[concept]),
                    1,
                    dtype=torch.int32,
                )
            )
        with open(file_concept_areas, "wb") as file:
            pickle.dump(areas, file)
    for i in range(len(areas)):
        if areas[i] is not None:
            areas[i] = areas[i].to(device)
    return areas


def get_overscribed_rectangles(masks, mask_shape):
    """
    Returns the vertices of the bounding box overscribed
    on the masks.

    Args:
        masks (torch.Tensor): tensor of masks.

    Returns:
        Tensor of vertices of the bounding box overscribed
        on the masks. The tensor has shape (num_samples, 2, 2).
        The first dimension corresponds to the sample index,
        the second and third dimension correspond to the top
        left and bottom right vertices of the bounding box,
        respectively.
    """
    num_samples = masks.shape[0]
    non_empty = torch.any(masks.reshape(num_samples, -1), 1)
    points = torch.tensor([[[0, 0], [0, 0]]] * num_samples)
    if non_empty.any():
        non_zero_points = torchvision.ops.masks_to_boxes(
            masks[non_empty].reshape(-1, mask_shape[0], mask_shape[1])
        )
        top_left = non_zero_points[:, 0:2]
        bottom_right = non_zero_points[:, 2:4]
        points[non_empty] = torch.tensor(
            list(zip(top_left.tolist(), bottom_right.tolist()))
        ).long()
    return points


def get_inscribed_rectangles(matrices, device):
    """Returns the vertices of the bounding box inscribed
    on the masks.

    Args:
        matrices (torch.Tensor): tensor of masks.

    Returns:
        Tensor of vertices of the bounding box inscribed
        on the masks. The tensor has shape (num_samples, 2, 2).
        The first dimension corresponds to the sample index,
        the second and third dimension correspond to the top
        left and bottom right vertices of the bounding box,
        respectively.
    """
    torch_num_samples = matrices.shape[0]
    torch_num_columns = matrices.shape[2]
    torch_num_rows = matrices.shape[1]

    # initialize left as the leftmost boundary possible
    torch_left = torch.zeros(torch_num_samples, torch_num_columns, device=device)
    # initialize right as the rightmost boundary possible
    torch_right = (
        torch.zeros(torch_num_samples, torch_num_columns, device=device)
        + torch_num_columns
    )
    torch_height = torch.zeros(torch_num_samples, torch_num_columns, device=device)
    torch_col_indices = torch.arange(torch_num_columns, device=device).repeat(
        torch_num_samples, 1
    )
    torch_max_area = torch.zeros(torch_num_samples, device=device)

    bottom_right_row = torch.zeros(torch_num_samples, device=device).int() - 1
    bottom_right_col = torch.zeros(torch_num_samples, device=device).int() - 1
    top_left_row = torch.zeros(torch_num_samples, device=device).int()
    top_left_col = torch.zeros(torch_num_samples, device=device).int()

    torch_col_indices = torch.arange(torch_num_columns, device=device).repeat(
        torch_num_samples, torch_num_rows, 1
    )

    # get indices of the closer zero from the left
    zero_indices_left = torch.zeros(
        torch_num_samples, torch_num_rows, torch_num_columns, device=device
    )
    for j in range(1, torch_num_columns):
        zero_indices_left[:, :, j] = torch.where(
            matrices[:, :, j - 1] == False,
            torch_col_indices[:, :, j],
            zero_indices_left[:, :, j - 1],
        )

    # get indices of the closer zero from the right
    zero_indices_right = (
        torch.zeros(torch_num_samples, torch_num_rows, torch_num_columns, device=device)
        + torch_num_columns
    )
    for j in range(torch_num_columns - 2, -1, -1):
        zero_indices_right[:, :, j] = torch.where(
            matrices[:, :, j + 1] == False,
            torch_col_indices[:, :, j + 1],
            zero_indices_right[:, :, j + 1],
        )

    for i in range(torch_num_rows):
        current_row = matrices[:, i, :]
        # update height
        torch_height = torch.where(current_row == True, torch_height + 1, 0)

        # update left
        torch_left = torch.where(
            current_row == True,
            torch.maximum(torch_left, zero_indices_left[:, i, :]),
            0,
        )

        # update right
        torch_right = torch.where(
            current_row == True,
            torch.minimum(torch_right, zero_indices_right[:, i, :]),
            torch_num_columns,
        )

        # update the area
        for j in range(torch_num_columns):
            area = torch_height[:, j] * (torch_right[:, j] - torch_left[:, j])
            bottom_right_col = torch.where(
                area > torch_max_area, torch_right[:, j] - 1, bottom_right_col
            )
            bottom_right_row = torch.where(area > torch_max_area, i, bottom_right_row)
            top_left_row = torch.where(
                area > torch_max_area,
                bottom_right_row - torch_height[:, j] + 1,
                top_left_row,
            )
            top_left_col = torch.where(
                area > torch_max_area,
                bottom_right_col - (torch_right[:, j] - torch_left[:, j]) + 1,
                top_left_col,
            )
            torch_max_area = torch.maximum(torch_max_area, area)

    top_left = torch.tensor(list(zip(top_left_row.tolist(), top_left_col.tolist())))
    bottom_right = torch.tensor(
        list(zip(bottom_right_row.tolist(), bottom_right_col.tolist()))
    )
    return torch.tensor(list(zip(top_left.tolist(), bottom_right.tolist())))


def get_concept_inscribed_masks(masks, mask_shape, info_directory, device):
    inscribed = []
    file_path = f"{info_directory}/positive_inscripted.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            inscribed = pickle.load(file)
    else:
        for concept in tqdm(
            range(len(masks)),
            total=len(masks),
            desc="Getting inscribed masks",
        ):
            concept_masks = general_utils.parse_mask_by_type(masks[concept])
            concept_masks = torch.reshape(
                concept_masks, (-1, mask_shape[0], mask_shape[1])
            ).to(device)
            inscribed.append(get_inscribed_rectangles(concept_masks, device).numpy())
        with open(file_path, "wb") as file:
            pickle.dump(inscribed, file)
    for i in range(len(inscribed)):
        inscribed[i] = torch.from_numpy(inscribed[i]).to(device)
    return inscribed


def get_bounding_boxes(masks, mask_shape, info_directory, device):
    """Returns the bounding boxes of the masks.

    Args:
        masks (list): list of masks.
        mask_shape (tuple): shape of a mask.
        info_directory (str): directory where to save/load the information.
        device (torch.device): device to use.

    Returns:
        list: list of bounding boxes of the masks.
    """
    overscribed = []
    file_path = f"{info_directory}/positive_rectangles.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            overscribed = pickle.load(file)
    else:
        for concept in tqdm(
            range(len(masks)),
            total=len(masks),
            desc="Getting bounding box for masks",
        ):
            concept_masks = general_utils.parse_mask_by_type(masks[concept])
            concept_masks = torch.reshape(
                concept_masks, (-1, mask_shape[0], mask_shape[1])
            )
            overscribed.append(
                get_overscribed_rectangles(concept_masks, mask_shape).numpy()
            )
        with open(file_path, "wb") as file:
            pickle.dump(overscribed, file)
    for i in range(len(overscribed)):
        overscribed[i] = torch.from_numpy(overscribed[i]).to(device)
    return overscribed


def get_rectangles_overlap(coordinates_a, coordinates_b, return_points=False):
    """Returns the overlap between two rectangles given their
    top left and bottom right coordinates"""
    a_top_left_x = coordinates_a[:, 0, 1]
    a_top_left_y = coordinates_a[:, 0, 0]
    a_bottom_right_x = coordinates_a[:, 1, 1] + 1
    a_bottom_right_y = coordinates_a[:, 1, 0] + 1
    b_top_left_x = coordinates_b[:, 0, 1]
    b_top_left_y = coordinates_b[:, 0, 0]
    b_bottom_right_x = coordinates_b[:, 1, 1] + 1
    b_bottom_right_y = coordinates_b[:, 1, 0] + 1
    x_overlap = torch.maximum(
        torch.zeros_like(a_bottom_right_x),
        torch.minimum(a_bottom_right_x, b_bottom_right_x)
        - torch.maximum(a_top_left_x, b_top_left_x),
    )
    y_overlap = torch.maximum(
        torch.zeros_like(a_bottom_right_y),
        torch.minimum(a_bottom_right_y, b_bottom_right_y)
        - torch.maximum(a_top_left_y, b_top_left_y),
    )
    overlap = x_overlap * y_overlap
    if return_points:
        # compute coordinates of the intersection rectangle
        top_left_x = torch.maximum(a_top_left_x, b_top_left_x)
        top_left_y = torch.maximum(a_top_left_y, b_top_left_y)
        bottom_right_x = torch.minimum(a_bottom_right_x, b_bottom_right_x) - 1
        bottom_right_y = torch.minimum(a_bottom_right_y, b_bottom_right_y) - 1
        return overlap, torch.tensor(
            list(
                zip(
                    torch.tensor(list(zip(top_left_y, top_left_x))),
                    torch.tensor(list(zip(bottom_right_y, bottom_right_x))),
                )
            )
        )
    else:
        return overlap
