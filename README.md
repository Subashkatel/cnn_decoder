# CNN decoder for planar color codes

Training-data generation and a neural pre-decoder for native 4.8.8 color-code
memories. The pipeline is detector inputs, local neural corrections, residual
detectors, then Tesseract or BP+LSD cleanup. This is a research adaptation of NVIDIA's
pre-decoder; a decoding improvement has not been established.

## Setup

Use Linux with Python 3.11. For GPU training on Della:

```bash
git clone https://github.com/Subashkatel/cnn_decoder.git
cd cnn_decoder
bash scripts/setup.sh --cuda
source .venv/bin/activate
```

The required Noah circuit builder and NVIDIA model and optimizer sources are
included in [`third_party/`](third_party/README.md), with original licenses and
file fingerprints. No original ZIP, separate source download, saved dataset,
or access to the original workspace is required.

Setup creates an isolated environment, installs the CUDA 12.4 build of PyTorch,
the pinned Python dependencies, and the compatible Tesseract wheel. It imports
the bundled sources without running a GPU workload. Set
`SETUP_PYTHON=/path/to/python3.11` when Python 3.11 is outside your PATH.

Submit the small GPU check:

```bash
sbatch scripts/smoke_test.sbatch
```

It exports a tiny dataset, performs two real GPU optimizer steps, saves and
reloads the checkpoint, and checks residual decoding with both backends.
Outputs stay under `.runtime/smoke/`. This checks installation and execution;
it does not establish a decoding improvement.

For a Linux machine where you explicitly want a CPU demonstration instead:

```bash
bash scripts/setup.sh --cpu
```

This runs the same tiny check on the CPU. The provided classical cleanup
implementations use one CPU. GPU training on Della runs only inside Slurm.

## Generate data

```bash
python -m training_data noah-training \
    --distance 3 --rounds 3 --shots 32 \
    --output .runtime/examples/planar_488_training
```

The export contains detector inputs, correction targets, masks, physical
trajectories, and logical labels. Generated files remain local and are ignored.
Use `python -m training_data noah-training --help` for the available options.

## Code and tests

- `training_data/`: detector handling, data generation, model, and training jobs.
- `third_party/`: the required upstream source, configuration, and licenses.
- `tests/`: contracts checked against Stim and deterministic injected faults.
- `scripts/`: setup and Della Slurm launchers.
- `verification/`: reusable policy selection, runtime analysis, and benchmarking tools.
- `detector_playground.ipynb`: a small detector tutorial with no saved outputs.

Start with the [planar reading guide](training_data/planar_488/README.md).

```bash
python -m unittest discover -s tests -v
```

Tests generate their small samples at runtime. No saved dataset is needed.

For the notebook, install a kernel in your environment:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name color-code-training-data \
    --display-name "Color code training data"
```

Select that kernel in VS Code. Clear notebook outputs before committing.

## GPU training on Della

Activate the environment and submit from the repository root:

```bash
python scripts/submit_planar_pilot.py
```

This submits GPU training followed by dependent CPU cleanup evaluation. The
scripts use the submission directory and active Python environment. They leave
QoS and CUDA device assignment to Slurm. See the planar guide for model profiles,
checkpoint continuation, and evaluation. Do not run GPU training on a login node.
Evaluation and threshold tuning accept `--cleanup bp_lsd` to use BP+LSD.

Keep datasets, checkpoints, downloaded bundles, job logs, experiment reports,
and machine-specific settings local. Work on separate branches and use pull
requests for source changes. After a public-history reset, use a fresh clone
so older artifacts are not merged back into the repository.

## Research sources

- [Shutty, Denser Planar Color Codes](https://arxiv.org/abs/2609.21376).
- [Olle et al., Fast and accurate AI-based pre-decoders for color codes](https://arxiv.org/abs/2607.10058).
