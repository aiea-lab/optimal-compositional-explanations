"""Script to run the clustering algorithm for compositional explanations."""

import copy

import torch
import absl.flags
import absl.app
import numpy as np

from src import settings
from utils import common_flags  # import all user flags
from utils import activations_utils, general_utils, compositional_utils

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
    beam_optimal_compo_exp = get_explanations(
        layer_activations=layer_activations,
        units=selected_units,
        cfg=cfg,
        heuristic="beam_optimal",
    )
    vanilla_compo_exp = get_explanations(
        layer_activations=layer_activations,
        units=selected_units,
        cfg=cfg,
        heuristic="none",
    )

    # Lists to computed individual heuristics stats
    mmesh_total_visited = []
    mmesh_total_expanded = []
    mmesh_total_estimated = []
    mmesh_total_time = []

    optimal_total_visited = []
    optimal_total_expanded = []
    optimal_total_estimated = []
    optimal_total_time = []

    beam_optimal_total_visited = []
    beam_optimal_total_expanded = []
    beam_optimal_total_estimated = []
    beam_optimal_total_time = []

    vanilla_total_visited = []
    vanilla_total_expanded = []
    vanilla_total_estimated = []
    vanilla_total_time = []

    optimal_ious = []
    beam_ious = []

    # Lists to compute comparison stats
    beam_iou_same_concepts_diff_iou = []
    optimal_iou_same_concepts_diff_iou = []
    beam_iou_diff_concepts_diff_iou = []
    optimal_iou_diff_concepts_diff_iou = []

    num_diff = 0
    total_meaningful = 0
    same_concepts_diff_iou = 0
    same_concepts_equal_iou = 0

    # Optimal Categories
    units_cat_1 = (
        []
    )  # different formula, different concepts, different iou (optimal better than mmesh)
    units_cat_2 = (
        []
    )  # different formula, same concepts, different iou (optimal better than mmesh)
    units_cat_3 = []  # different formula, same concepts, same iou

    for index in range(len(selected_units)):
        (
            unit,
            cluster_index,
            _,
            mmesh_label,
            mmesh_string_label,
            mmesh_iou,
            mmesh_visited,
            mmesh_expanded,
            mmesh_estimated,
            mmesh_time_taken,
        ) = mmesh_compo_exp[index]
        # Individual Stats when available
        if mmesh_label is not None:
            mmesh_total_visited.append(mmesh_visited)
            mmesh_total_expanded.append(mmesh_expanded)
            mmesh_total_estimated.append(mmesh_estimated)
            mmesh_total_time.append(mmesh_time_taken)
        (
            _,
            _,
            _,
            optimal_label,
            optimal_string_label,
            optimal_iou,
            optimal_visited,
            optimal_expanded,
            optimal_estimated,
            optimal_time_taken,
        ) = optimal_compo_exp[index]
        if optimal_label is not None:
            optimal_total_visited.append(optimal_visited)
            optimal_total_expanded.append(optimal_expanded)
            optimal_total_estimated.append(optimal_estimated)
            optimal_total_time.append(optimal_time_taken)
        (
            _,
            _,
            _,
            beam_optimal_label,
            beam_optimal_string_label,
            beam_optimal_iou,
            beam_optimal_visited,
            beam_optimal_expanded,
            beam_optimal_estimated,
            beam_optimal_time_taken,
        ) = beam_optimal_compo_exp[index]
        if beam_optimal_label is not None:
            beam_optimal_total_visited.append(beam_optimal_visited)
            beam_optimal_total_expanded.append(beam_optimal_expanded)
            beam_optimal_total_estimated.append(beam_optimal_estimated)
            beam_optimal_total_time.append(beam_optimal_time_taken)
        (
            _,
            _,
            _,
            vanilla_label,
            vanilla_string_label,
            vanilla_iou,
            vanilla_visited,
            vanilla_expanded,
            vanilla_estimated,
            vanilla_time_taken,
        ) = vanilla_compo_exp[index]
        if vanilla_label is not None:
            vanilla_total_visited.append(vanilla_visited)
            vanilla_total_expanded.append(vanilla_expanded)
            vanilla_total_estimated.append(vanilla_estimated)
            vanilla_total_time.append(vanilla_time_taken)

        beam_label = (
            mmesh_label
            if mmesh_label
            else (beam_optimal_label if beam_optimal_label else vanilla_label)
        )
        beam_iou = (
            mmesh_iou
            if mmesh_label
            else (beam_optimal_iou if beam_optimal_label else vanilla_iou)
        )
        beam_string_label = (
            mmesh_string_label
            if mmesh_label
            else (
                beam_optimal_string_label
                if beam_optimal_label
                else vanilla_string_label
            )
        )

        # Comparison when possible (computed over the same units)
        if beam_label is not None and optimal_label is not None:
            total_meaningful += 1
            if beam_label != optimal_label:
                atoms_beam = beam_label.get_vals()
                atoms_optimal = optimal_label.get_vals()
                if set(atoms_beam) == set(atoms_optimal):
                    if optimal_iou != beam_iou:
                        same_concepts_diff_iou += 1
                        beam_iou_same_concepts_diff_iou.append(beam_iou)
                        optimal_iou_same_concepts_diff_iou.append(optimal_iou)
                        units_cat_2.append(unit)
                    else:
                        same_concepts_equal_iou += 1
                        units_cat_3.append(unit)
                    print(
                        f"Same Concepts different formula for Unit {unit} Cluster {cluster_index} - Beam: {beam_string_label} {mmesh_label} ({mmesh_iou}) - Optimal: {optimal_string_label} {optimal_label} ({optimal_iou})"
                    )
                elif optimal_iou == beam_iou:
                    print(
                        f"Same IoU different concepts for Unit {unit} Cluster {cluster_index} - Beam: {beam_string_label} {mmesh_label} ({mmesh_iou}) - Optimal: {optimal_string_label} {optimal_label} ({optimal_iou})"
                    )
                    print(f"This behavior has never been observed in our experiments")
                elif optimal_iou > beam_iou:
                    beam_iou_diff_concepts_diff_iou.append(beam_iou)
                    optimal_iou_diff_concepts_diff_iou.append(optimal_iou)
                    units_cat_1.append(unit)

                beam_ious.append(beam_iou)
                optimal_ious.append(optimal_iou)
                print(f"Unit {unit}")
                print(f"Beam: {beam_string_label} ({beam_iou:<.8f})")
                print(f"Optimal: {optimal_string_label} ({optimal_iou:<.8f})")
                print()

                num_diff += 1

    # PRINT SUMMARY STATS
    print(f"Number of different results: {num_diff} out of {total_meaningful} units")
    if len(beam_ious) > 0 and len(optimal_ious) > 0:
        print(
            f"Average Beam IoU over {len(beam_ious)}: {np.mean(beam_ious):.4f} - Standard Deviation: {np.std(beam_ious):.4f}"
        )
        print(
            f"Average Optimal IoU over {len(optimal_ious)}: {np.mean(optimal_ious):.4f} - Standard Deviation: {np.std(optimal_ious):.4f}"
        )
        print(f"Category 1: Different formula, different concepts, different iou")
        print(
            f"\t: {(len(optimal_ious)- same_concepts_diff_iou - same_concepts_equal_iou)/len(optimal_ious)}"
        )
        print(f"Units with different formulas and ious: {units_cat_1}")
        print(
            f"IoU different formula different concepts - Beam Average IoU over {len(beam_iou_diff_concepts_diff_iou)}: {np.mean(beam_iou_diff_concepts_diff_iou):.4f} - Standard Deviation: {np.std(beam_iou_diff_concepts_diff_iou):.4f}"
        )
        print(
            f"IoU different formula different concepts - Optimal Average IoU over {len(optimal_iou_diff_concepts_diff_iou)}: {np.mean(optimal_iou_diff_concepts_diff_iou):.4f} - Standard Deviation: {np.std(optimal_iou_diff_concepts_diff_iou):.4f}"
        )
        print(f"Category 2: Different formula, same concepts, different iou")
        print(f"\t Percentage: {len(units_cat_2)/len(optimal_ious)}")
        print(f"\t Units: {units_cat_2}")
        print(
            f"IoU same concepts different iou - Beam Average IoU over {len(beam_iou_same_concepts_diff_iou)}: {np.mean(beam_iou_same_concepts_diff_iou):.4f} - Standard Deviation: {np.std(beam_iou_same_concepts_diff_iou):.4f}"
        )
        print(
            f"IoU same concepts different iou - Optimal Average IoU over {len(optimal_iou_same_concepts_diff_iou)}: {np.mean(optimal_iou_same_concepts_diff_iou):.4f} - Standard Deviation: {np.std(optimal_iou_same_concepts_diff_iou):.4f}"
        )
        print(
            f"Number of times C same concepts same iou and different formula: {same_concepts_equal_iou/len(optimal_ious)}"
        )
        print(f"Units with same concepts same iou and different formula: {units_cat_3}")
        print(
            f"Units where different from optimal: {units_cat_1+units_cat_2+units_cat_3}"
        )
    print(f"Total Units: {len(selected_units)}")
    print(
        f"Optimal Average Visited over {len(optimal_total_visited)}: {np.mean(optimal_total_visited):.2f} - Standard Deviation: {np.std(optimal_total_visited):.2f}"
    )
    print(
        f"Optimal Average Expanded over {len(optimal_total_expanded)}: {np.mean(optimal_total_expanded):.2f} - Standard Deviation: {np.std(optimal_total_expanded):.2f}"
    )
    print(
        f"Optimal Average Estimated over {len(optimal_total_estimated)}: {np.mean(optimal_total_estimated):.2f} - Standard Deviation: {np.std(optimal_total_estimated):.2f}"
    )
    print(
        f"Optimal Average Time over {len(optimal_total_time)}: {np.mean(optimal_total_time):.2f} seconds - Standard Deviation: {np.std(optimal_total_time):.2f} seconds"
    )
    print(
        f"Beam Optimal Average Visited over {len(beam_optimal_total_visited)}: {np.mean(beam_optimal_total_visited):.2f} - Standard Deviation: {np.std(beam_optimal_total_visited):.2f}"
    )
    print(
        f"Beam Optimal Average Expanded over {len(beam_optimal_total_expanded)}: {np.mean(beam_optimal_total_expanded):.2f} - Standard Deviation: {np.std(beam_optimal_total_expanded):.2f}"
    )
    print(
        f"Beam Optimal Average Estimated over {len(beam_optimal_total_estimated)}: {np.mean(beam_optimal_total_estimated):.2f} - Standard Deviation: {np.std(beam_optimal_total_estimated):.2f}"
    )
    print(
        f"Beam Optimal Average Time over {len(beam_optimal_total_time)}: {np.mean(beam_optimal_total_time):.2f} seconds - Standard Deviation: {np.std(beam_optimal_total_time):.2f} seconds"
    )
    print(
        f"M-MESH Average Visited over {len(mmesh_total_visited)}: {np.mean(mmesh_total_visited):.2f} - Standard Deviation: {np.std(mmesh_total_visited):.2f}"
    )
    print(
        f"M-MESH Average Expanded over {len(mmesh_total_expanded)}: {np.mean(mmesh_total_expanded):.2f} - Standard Deviation: {np.std(mmesh_total_expanded):.2f}"
    )
    print(
        f"M-MESH Average Estimated over {len(mmesh_total_estimated)}: {np.mean(mmesh_total_estimated):.2f} - Standard Deviation: {np.std(mmesh_total_estimated):.2f}"
    )
    print(
        f"M-MESH Average Time over {len(mmesh_total_time)}: {np.mean(mmesh_total_time):.2f} seconds - Standard Deviation: {np.std(mmesh_total_time):.2f} seconds"
    )
    print(
        f"Vanilla Average Visited over {len(vanilla_total_visited)}: {np.mean(vanilla_total_visited):.2f} - Standard Deviation: {np.std(vanilla_total_visited):.2f}"
    )
    print(
        f"Vanilla Average Expanded over {len(vanilla_total_expanded)}: {np.mean(vanilla_total_expanded):.2f} - Standard Deviation: {np.std(vanilla_total_expanded):.2f}"
    )
    print(
        f"Vanilla Average Estimated over {len(vanilla_total_estimated)}: {np.mean(vanilla_total_estimated):.2f} - Standard Deviation: {np.std(vanilla_total_estimated):.2f}"
    )
    print(
        f"Vanilla Average Time over {len(vanilla_total_time)}: {np.mean(vanilla_total_time):.2f} seconds - Standard Deviation: {np.std(vanilla_total_time):.2f} seconds"
    )


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
