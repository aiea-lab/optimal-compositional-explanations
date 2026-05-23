
# Reproducing Paper results

This document provides instructions for reproducing the results in the paper.

## Setup Repo 

### Step 1: Create environment. 

The suggested option for creating the environment is to use Docker.

Pull the image `nvcr.io/nvidia/pytorch:23.01-py3` (<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-01.html">here</a>) from nvidia NGC, which contains all the dependencies needed to run the code. 

Then, run the container.

If you use Detectron2-backed datasets (e.g., Cityscapes/ADE20K registration), also install Detectron2 within the container:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

As an alternative to Docker, you can also create a Python virtual environment and install the dependencies to replicate the nvcr.io/nvidia/pytorch:23.01-py3 image. 
 

### Step 2: Download Required Assets

From the repository root:

```bash
bash scripts/download_models.sh
```

This downloads all probed models.

To run experiments on Broden:

```bash
bash scripts/download_broden.sh
```

This will populate:

- `data/model/zoo/` with Places365 checkpoints
- `data/dataset/broden1_224` and `data/dataset/broden1_227`

## Step 3: Set Detectron2 Variable
Detectron2 datasets require the `DETECTRON2_DATASETS` variable to be set to the directory where the datasets are stored.

```bash
export DETECTRON2_DATASETS=<DIR>
```

## Step 4: Run scripts 

**Note**: The first time you run `run_clustering.py`, it will generate and save activations, concept masks for all concepts in the dataset, and heuristic-specific cached information. This can take a long time, especially for Broden. However, once masks and heuristic information are cached, subsequent runs on the same dataset will load these precomputed values and be significantly faster. Saving masks can take up to 1-2 hours for Broden. Generating heuristic information can take between 2 and 4 hours, depending on the heuristic.

### Numerical Results
These are the main commands needed to reproduce the paper's core numerical results.

For each of these, first run `run_clustering.py` with the appropriate flags, then run `compare_methods.py` with the exact same flags to get summary statistics.

### Table 2 to Table 7

To run experiments for Tables 2-7:
```bash
python3 run_clustering.py \
	--dataset=<DATASET> \
	--heuristic=<ALGORITHM> \
	--num_clusters=1 \
	--random_units=50
```
Replace \<DATASET> and \<ALGORITHM> with the appropriate values:

(1) \<DATASET>: use `cityscapes_fine_sem_seg_val` for low complexity, `ade20k_full_sem_seg_freq_val_all` for intermediate complexity, or `broden` for the highest complexity setting.

(2) \<ALGORITHM>: use `optimal`, `beam_optimal`, `mmesh`, or `none` to run the optimal algorithm, the beam variant using our sample heuristic, M-MESH, or vanilla beam search, respectively.

### Table 8

To run the experiments for Table 8:
```bash
python3 run_clustering.py \
	--dataset='ade20k_full_sem_seg_freq_val_all' \
	--heuristic=<ALGORITHM> \
	--num_clusters=1 \
	--random_units=50 \
	--length=<LENGTH> \
	--beam_limit=<BEAM_SIZE>
```
Replace \<ALGORITHM>, \<LENGTH>, and \<BEAM_SIZE> as follows:

(1) \<ALGORITHM>: `beam_optimal` or `mmesh` (for the first and second columns, respectively).
(2) \<LENGTH>: `3`, `5`, `10`, or `20` (while keeping beam size = 5).
(3) \<BEAM_SIZE>: `5`, `10`, or `20` (while keeping length = 3).

### Table 9

To run the experiment for Table 9:
```bash
python3 run_clustering.py \
	--dataset='broden' \
	--heuristic='optimal' \
	--num_clusters=5 \
	--random_units=50 \
```


### Images

To generate the images included in the appendix, run:
```bash
python3 run_clustering.py \
	--dataset='broden' \
	--heuristic='optimal' \
	--num_clusters=1 \
```

and
```bash
python3 run_clustering.py \
	--dataset='broden' \
	--heuristic='mmesh' \
	--num_clusters=1 \
```
Then generate the images with:

```bash
python3 generate_images.py \
	--dataset='broden' \
	--num_clusters=1 \
```

### Deterministic CUDA Behavior
Use deterministic CUDA behavior (recommended for reproducibility):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_clustering.py \
	--dataset=broden \
	--heuristic=optimal \
	--num_clusters=1

```

## CLI Flags

Common flags are defined in `utils/common_flags.py`.

- `--dataset`: default `cityscapes_fine_sem_seg_val`
- `--model`: default `resnet18` (also supports `alexnet`, `densenet161`)
- `--layer`: default `layer4` (for `alexnet` and `densenet161` use `features`)
- `--heuristic`: `optimal`, `beam_optimal`, `mmesh`, `none`
- `--length`: explanation length (default `3`)
- `--beam_limit`: beam size (default `5`)
- `--units` / `--random_units`: unit selection
- `--device`: compute device (default `cuda`)


## Reproducibility Notes

- Set `--seed` (default `0`) for deterministic behavior where possible (this applies mainly if you are using more than 1 cluster).
- Use deterministic CUDA behavior (recommended for reproducibility):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_clustering.py
```
- Larger concept spaces can be memory-intensive. Consider changing the following flags to reduce memory usage:
1. `--preload_masks`
2. `--step_size`
3. `--fast_impl`


## Changes after ICML
**Note**: After submission, we applied several optimizations to make the code run faster. These edits do not change the meaning or scale of the results. However, small variations may appear in visited-node counts and runtime. Specifically, we made the following improvements:

- We unified the implementation of all beam variants so they now share almost the same structure (except for state-space ordering).
- We updated the formula implementation:
	- We replaced the old formula hash function (available <a href="https://github.com/KRLGroup/Clustered-Compositional-Explanations/blob/main/src/formula.py">here</a>) with a new one that is slightly faster, although it covers fewer cases than the previous implementation.
	- We removed the `OrderedFormula` class (described <a href="https://github.com/KRLGroup/Clustered-Compositional-Explanations/blob/main/src/formula.py">here</a>) and now handle both queueing and ordering directly with the formula class (previously, as in CCE, `OrderedFormula` was used both to sort the state space and as the heap key during optimal search).
- We optimized propagation:
	- The code now treats every newly discovered ancestor during propagation as a visited node (previously, only nodes expanded during search were considered visited; ancestors could therefore be revisited multiple times, and IoU generation for them during propagation was not counted as state visitation).
	- During propagation, we now generate quantities and propagate information only for states that have not already been visited (previously this was done every time).
- For beam variants, we no longer expand nodes in the current beam that were already expanded in previous beam levels.


We tested this implementation and we didn't observe significant differences in the results compared to the previous implementation. If you observe any significant differences, please let us know by opening an issue. We maintain the legacy code and we would be happy to investigate any discrepancies.


## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{
LaRosa2026,
title={Guaranteed Optimal Compositional Explanations for Neurons},
author={Biagio {La Rosa} and Leilani H. Gilpin},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=MHiiwC3oFR}
}
```

