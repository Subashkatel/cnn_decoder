# CNN decoder for planar color codes

Training-data generation and a neural pre-decoder for native 4.8.8 color-code
memories. The pipeline is detector inputs, local neural corrections, residual
detectors, then Tesseract cleanup. This is a research adaptation of NVIDIA's
pre-decoder; a decoding improvement has not been established.

## Setup

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the original source dependencies locally:

```bash
git clone https://github.com/NVIDIA/Ising-Decoding.git Ising-Decoding
git -C Ising-Decoding checkout --detach ea7c1bb3b823fc6a38472d0d95974b6d01932288
```

Obtain Noah's source bundle from the
[paper's release record](https://doi.org/10.5281/zenodo.22820614) and extract it
locally with this layout:

```text
denser_planar_color_codes_source_v1/
  src/denser_codes.py
  src/vendor/
  data/decoder_configs.json
```

The source directories and downloaded archives are ignored by Git. Only the
adapter code is maintained here. Generated reports identify the installed Noah
source by a source-tree fingerprint, so an original ZIP is not required at
runtime.

For the Linux cleanup decoder and decoder-dependent tests:

```bash
bash scripts/setup_tesseract.sh
```

The script uses the active Python interpreter. Set `TRAINING_PYTHON` to an
interpreter path to use a different environment.

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

Keep datasets, checkpoints, downloaded bundles, job logs, experiment reports,
and machine-specific settings local. Work on separate branches and use pull
requests for source changes. After a public-history reset, use a fresh clone
so older artifacts are not merged back into the repository.

## Research sources

- [Shutty, Denser Planar Color Codes](https://arxiv.org/abs/2609.21376).
- [Olle et al., Fast and accurate AI-based pre-decoders for color codes](https://arxiv.org/abs/2607.10058).
