"""Shared command-line flags used across experiment scripts.

Defines dataset, model, search, reproducibility, and path parameters.
"""

import absl.flags

import datasets  # register all new datasets

# Reproducibility parameters
absl.flags.DEFINE_integer("seed", 0, "seed to use to set reproducibility")

# General parameters
absl.flags.DEFINE_string(
    "dataset",
    "cityscapes_fine_sem_seg_val",
    "subset to use. Values:[cityscapes_fine_sem_seg_val,ade20k_full_sem_seg_freq_val_all, broden]",
)
absl.flags.DEFINE_string(
    "model", "resnet18", "model to use. Values:[resnet18, alexnet, densenet161]"
)
absl.flags.DEFINE_string("layer", "layer4", "Layer to analyze.")
absl.flags.DEFINE_list("units", None, "Specific units to investigate")
absl.flags.DEFINE_integer("random_units", 0, "number of units")

# Paths parameters
absl.flags.DEFINE_string("root_datasets", "datasets/data", "root directory for datasets")
absl.flags.DEFINE_string(
    "root_segmentations", "data/cache/segmentations", "root directory for segmentations"
)
absl.flags.DEFINE_string(
    "root_activations", "data/cache/activations", "root directory for activations"
)
absl.flags.DEFINE_string("root_results", "data/results", "root directory for results")
absl.flags.DEFINE_string(
    "root_optimal_info", "data/cache/optimal_info", "directory to store optimal info"
)
absl.flags.DEFINE_string("root_models", "data/model", "root directory for models")

# Explanation parameters
absl.flags.DEFINE_integer("num_clusters", 1, "number of clusters")
absl.flags.DEFINE_string("heuristic", "optimal", "heuristic to use")
absl.flags.DEFINE_integer("length", 3, "length of explanations")


# Explanation Beam Parameters
absl.flags.DEFINE_integer("beam_limit", 5, "beam limit")

# Hardware parameters
absl.flags.DEFINE_string("device", "cuda", "device to use to store the model")
absl.flags.DEFINE_boolean(
    "preload_masks",
    True,
    "whether to load segmentation masks in memory or to load them on the fly during the search",
)
absl.flags.DEFINE_integer(
    "step_size",
    60,
    "step size to use to compute the masks. If None, all concepts are computed at once. If not None, the concepts are computed in batches of size step_size. This is useful to reduce the memory usage when the number of concepts is large.",
)
absl.flags.DEFINE_boolean(
    "fast_impl",
    False,
    "whether to use the fast implementation to compute the masks. The fast implementation preload the whole datasaetsegmentations and compute the masks using the segmentations, while the slow implementation compute the masks from batches and repeat the dataset parsing for each step size. The fast implementation is faster but requires more memory, while the slow implementation is slower but requires less memory.",
)
