"""Dataset construction and adapter utilities for supported benchmarks.

Provides wrappers to normalize image and segmentation access patterns.
"""

import torch
import torchvision.transforms
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper

from . import general_utils
from . import segmentations


def get_probing_transformations(img_size, mean, std):
    """Returns the transformation compatible with probing.
    Args:
        img_size (int): size of the image to be transformed
        mean (list): mean of the dataset to be transformed
        std (list): standard deviation of the dataset to be transformed
    Returns:
        transformation (torchvision.transforms.Compose): transformation compatible with probing
    """
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def get_dataset(cfg):
    """Returns the dataset object for the given dataset name.
    Args:
        cfg (Settings): settings object containing the dataset name and other parameters
    Returns:
        dataset (DatasetWrapper): dataset object for the given dataset name
    """
    dataset_name = cfg.get_dataset_name().lower()
    if dataset_name in DatasetCatalog:
        return DatasetCatalog.get(dataset_name)
    elif dataset_name == "broden":
        return segmentations.BrodenDataset(
            cfg.get_root_datasets(),
            resolution=cfg.get_img_size(),
            broden_version=1,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not suuported yet")


class DatasetWrapper:

    def __init__(self):
        raise NotImplementedError

    def extract_segmentations(self):
        raise NotImplementedError


class DetectronWrapper(DatasetWrapper):
    """Dataset class for detectron2 datasets. It is used to load the images and the annotations from the detectron2 datasets. It is used to compute the segmentation masks for the detectron2 datasets."""

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name in DatasetCatalog:
            self.dataset = DatasetCatalog.get(dataset_name)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not suuported yet")
        self.concept_labels = MetadataCatalog.get(dataset_name).stuff_classes.copy()
        self.data_loader = build_detection_test_loader(
            self.dataset,
            mapper=DatasetMapper(
                is_train=False,
                augmentations=[],
                image_format="RGB",
            ),
            batch_size=1,
        )

        self.data_iter_fn = lambda x: x[0][
            "image"
        ]  # For detectron2 datasets, the data loader returns a list of dictionaries, where each dictionary contains the image and the annotations. We need to extract the image from the dictionary to be compatible with the probing code.

    def extract_segmentations(self, data):
        """Extracts the segmentation masks from the data returned by the data loader.
        Args:
            data (list): list of dictionaries returned by the data loader, where each dictionary contains the image and the annotations.
        Returns:
            segmentation_masks (torch.Tensor): tensor of shape (1, 1, H, W) containing the segmentation masks for the image, where H and W are the height and width of the image, respectively.
        """
        return data[0]["sem_seg"].unsqueeze(0).unsqueeze(0)


class BrodenWrapper(DatasetWrapper):
    """Dataset class for broden dataset. It is used to load the images and the annotations from the broden dataset. It is used to compute the segmentation masks for the broden dataset."""

    def __init__(self, config):
        self.cfg = config
        self.dataset = segmentations.BrodenDataset(
            config.get_root_datasets(),
            resolution=config.get_img_size(),
            broden_version=1,
            transform_image=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        seed = config.get_seed()
        self.concept_labels = self.dataset.labels
        generator = general_utils.set_seed(seed)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            worker_init_fn=general_utils.seed_worker,
            generator=generator,
        )

        self.data_iter_fn = lambda x: x[
            0
        ]  # For broden dataset, the data loader returns a list of tuples, where each tuple contains the image and the annotations. We need to extract the image from the tuple to be compatible with the probing code.

    def extract_segmentations(self, data):
        """Extracts the segmentation masks from the data returned by the data loader.
        Args:
            data (tuple): tuple containing the image and the annotations returned by the data loader, where the first element of the tuple is the image and the second element is the annotations.
        Returns:
            segmentation_masks (torch.Tensor): tensor of shape (1, 1, H, W) containing the segmentation masks for the image, where H and W are the height and width of the image, respectively.
        """
        return data[1]
