"""Script to run the clustering algorithm for compositional explanations."""

import torch
import absl.flags
import absl.app

from src import settings
from utils import common_flags  # import all user flags
from utils import activations_utils, general_utils, mask_utils, compositional_utils

# Flags specific to this script
absl.flags.DEFINE_string("output_path", "output/explanations", "output path")

FLAGS = absl.flags.FLAGS


def main(argv):
    if FLAGS.num_clusters < 1:
        raise ValueError("num_clusters must be greater than 0")

    # Set seed
    general_utils.set_seed(FLAGS.seed)

    # Parameters
    cfg = settings.Settings(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        num_clusters=FLAGS.num_clusters,
        beam_limit=FLAGS.beam_limit,
        heuristic=FLAGS.heuristic,
        length=FLAGS.length,
        layer=FLAGS.layer,
        device=FLAGS.device,
        seed=FLAGS.seed,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
    )

    # Load Activations
    print(
        f"Loading activations for {cfg.get_model()} model on {cfg.get_dataset_name()} dataset - Layer:{cfg.get_layer()}"
    )
    layer_activations = activations_utils.get_layer_activations(cfg)
    print(f"Activation Loaded")

    # Select Units
    selected_units = activations_utils.get_selected_units(
        layer_activations, units=FLAGS.units, random_units=FLAGS.random_units
    )
    selected_units = sorted(selected_units)
    if FLAGS.preload_masks:
        masks, masks_labels = mask_utils.get_masks(cfg)
    else:
        masks, masks_labels = mask_utils.load_masks_path(cfg)
        print(
            f"Masks Not Pre-Loaded - Masks will be loaded during the heuristic search"
        )

    # Get Info
    disjoint_info = mask_utils.get_disjoint_info(cfg)
    if FLAGS.heuristic == "optimal" or FLAGS.heuristic == "beam_optimal":
        masks_info = mask_utils.get_masks_info(masks, config=cfg, quantities=True)
    elif FLAGS.heuristic == "mmesh":
        masks_info = mask_utils.get_masks_info(
            masks,
            config=cfg,
            areas=True,
            quantities=False,
            inscribed=True,
            bb_boxes=True,
        )
    elif FLAGS.heuristic == "none":
        masks_info = None
    else:
        raise ValueError(f"Unknown heuristic {FLAGS.heuristic}")

    # Compute Compositional Explanations
    print("Number of concepts: ", len(masks))
    print("Computing Compositional Explanations")

    compositional_utils.compute_compositional_explanations(
        masks=masks,
        masks_labels=masks_labels,
        masks_info=masks_info,
        disjoint_info=disjoint_info,
        activations=layer_activations,
        units=selected_units,
        config=cfg,
    )


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
