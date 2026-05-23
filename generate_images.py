"""Script to run the clustering algorithm for compositional explanations."""

import copy
import os

import torch
import absl.flags
import absl.app

from compositional import formula as F
from src import settings
from utils import common_flags  # import all user flags
from utils import (
    activations_utils,
    general_utils,
    mask_utils,
    compositional_utils,
    dataset_utils,
)

# Flags specific to this script
absl.flags.DEFINE_string(
    "figures_path", "output/explanations", "output path for the figures"
)

FLAGS = absl.flags.FLAGS


def get_explanations(*, layer_activations, units, cfg, heuristic):
    # make deep copy of cfg and add heuristic to it
    heuristic_cfg = copy.deepcopy(cfg)
    heuristic_cfg.set_heuristic(heuristic)
    h_exp = compositional_utils.load_compositional_explanations(
        activations=layer_activations, units=units, config=heuristic_cfg
    )
    return h_exp


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
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
    )

    mask_shape = cfg.get_mask_shape()

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

    # Load labels and masks
    masks, concept_labels = mask_utils.get_masks(cfg)

    # Load dataset images
    dataset = dataset_utils.get_dataset(cfg=cfg)

    # Load Explanations from different heuristics
    mmesh_compo_exp = get_explanations(
        layer_activations=layer_activations,
        units=selected_units,
        cfg=cfg,
        heuristic="mmesh",
    )
    optimal_compo_exp = get_explanations(
        layer_activations=layer_activations,
        units=selected_units,
        cfg=cfg,
        heuristic="optimal",
    )

    pathdir = f"{FLAGS.figures_path}/{cfg.get_model()}/"
    if os.path.exists(pathdir) == False:
        os.makedirs(pathdir)
    print(f"Saving figures in {pathdir}")
    for index in range(len(selected_units)):
        (
            unit,
            _,
            mmesh_activation_range,
            mmesh_label,
            mmesh_string_label,
            mmesh_best_iou,
            _,
            _,
            _,
            _,
        ) = mmesh_compo_exp[index]
        unit, _, _, optimal_label, optimal_string_label, optimal_iou, _, _, _, _ = (
            optimal_compo_exp[index]
        )

        if (
            mmesh_label is not None
            and optimal_label is not None
            and mmesh_label != optimal_label
        ):
            # Get bitmaps
            unit_activations = layer_activations[:, unit, :, :]
            bitmaps = activations_utils.compute_bitmaps(
                unit_activations,
                mmesh_activation_range,
                mask_shape=mask_shape,
            )
            device = (
                torch.device(cfg.get_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            bitmaps = bitmaps.to(device)
            number_samples = 4

            # Get string explanations
            mmesh_str_best_label = F.get_formula_str(mmesh_label, concept_labels)
            optimal_str_best_label = F.get_formula_str(optimal_label, concept_labels)

            # Generate Samples Grid
            mmesh_grid = general_utils.get_grid_intersection(
                dataset=dataset,
                label=mmesh_label,
                masks=masks,
                bitmaps=bitmaps,
                number_samples=number_samples,
                mask_shape=mask_shape,
                device=cfg.get_device(),
            )
            optimal_grid = general_utils.get_grid_intersection(
                dataset=dataset,
                label=optimal_label,
                masks=masks,
                bitmaps=bitmaps,
                number_samples=number_samples,
                mask_shape=mask_shape,
                device=cfg.get_device(),
            )

            # Save figure
            figure = general_utils.get_figure(
                [optimal_grid, mmesh_grid],
                [
                    f"Unit:{unit} \n Optimal: {optimal_str_best_label} | IoU:{round(optimal_iou,4)}",
                    f"Beam: {mmesh_str_best_label} | IoU:{round(mmesh_best_iou,4)}",
                ],
            )
            figure.savefig(f"{pathdir}/" + f"{unit}.png")

            print(f"Unit {unit}")
            print(f"M-MESH: {mmesh_string_label} ({mmesh_best_iou:<.4f})")
            print(f"Optimal: {optimal_string_label} ({optimal_iou:<.4f})")
            print()


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
