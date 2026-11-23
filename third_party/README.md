# Required upstream source

These files are included so a checkout contains the circuit builder, neural
architecture, and optimizer needed by this project. No separate source download
is required. Python packages are installed by `scripts/setup.sh`.

- `noah/`: the unchanged planar 4.8.8 constructor and its dependencies from
  Noah Shutty's supplied `denser_planar_color_codes_source_v1` bundle for
  [Denser Planar Color Codes](https://arxiv.org/abs/2609.21376).
- `nvidia/`: the unchanged model and Lion optimizer, including their import
  dependencies, from [NVIDIA Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding)
  revision `ea7c1bb3b823fc6a38472d0d95974b6d01932288`.

Original licenses and notices accompany each subset. `sources.toml` records the
original path and SHA256 of each copied file. The source integrity test checks
these fingerprints. Keep changes to our adaptation in `training_data/`.

The two included JSON files are required configuration: Noah's local circuit
construction rules and recorded decoder settings. No sampled detector data,
paper simulation results, model weights, or downloaded archives are included.
The other code families offered by the original Noah bundle are outside this
project's supported subset.
