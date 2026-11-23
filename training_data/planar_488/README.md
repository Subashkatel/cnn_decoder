# Planar 4.8.8 training data

Our adaptation uses Noah's unchanged native-CZ SI1000 circuits and NVIDIA's
configurable Conv3D model. The current format is
`planar_488_local_targets_v4` with `compact_data_anchors_v1` embedding.
New runs use the `round_local` target protocol. Existing v3 checkpoints retain
their `trajectory` protocol; a checkpoint cannot cross label versions.
Older v2 checkpoints and tensors are incompatible and are rejected when loading.
This is an adaptation, not a reproduction of NVIDIA's complete training recipe.

## Read the code in this order

1. [`generator.py`](generator.py): `TrainingCase` caches deterministic setup;
   `sample` generates fresh shots and verifies the encoded targets.
2. [`code.py`](code.py): native qubit order, CSS algebra, and record-map identities.
3. [`trajectory.py`](trajectory.py): correlated noise, boundary frames, and
   removal of Hadamard coordinates and record-dependent Pauli byproducts.
4. [`targets.py`](targets.py): spatial reduction, temporal correction labels,
   and which targets are defined.
5. [`tensors.py`](tensors.py), [`channels.py`](channels.py): compact anchors,
   encoding and decoding of heads, named channels, and the X/Z axis order.
6. [`model.py`](model.py), [`learning.py`](learning.py): model, masked
   loss, optimizer step, and full checkpoint continuation.
7. [`jobs/train.py`](jobs/train.py): GPU training and validation-only model selection.
8. [`evaluation.py`](evaluation.py), [`cleanup_decoder.py`](cleanup_decoder.py),
   [`jobs/evaluate.py`](jobs/evaluate.py): correction parities and resumable
   CPU cleanup evaluation.

`jobs/` holds worker entry points and their shared configuration and runtime helpers.
`jobs/configuration.py` owns the run settings and the training report that both
jobs read. `jobs/runtime.py` owns allocation checks, provenance, and
signal/time handling. `jobs/check_gpu.py` is the short CUDA check.
`dependencies.py` loads the bundled Noah circuit builder and NVIDIA model from
`third_party/`, and the compatible project-local Tesseract package. BP+LSD uses
`bp_cleanup.py`; both backends retain every native flag detector.
`training_data/reports.py` owns atomic JSON publication and content hashes.

## Frames, labels, and the opening reference

`Trajectory.frames` uses `(shot, boundary, X/Z, data_qubit)` axes. Boundaries
are captured after each ancilla-reset moment, including its data wait noise.
The final frame is captured before the final data measurement, after closing
rotations and premeasurement noise. This places reset-wait errors at the
boundary whose detector sees them.

The simulator jointly samples physical Pauli channels. Stabilizer randomization
is disabled to choose a fixed frame gauge. `measurement_flips` are noise flips
relative to a noiseless reference, not full physical measurement outcomes.
Hadamard coordinates and accumulated record byproducts are removed from the
physical frames. A second simulator draws the same faults and propagates each
interval's new faults separately. Only this auxiliary simulator's data frame is
cleared after each input-boundary snapshot; physical detector and logical
samples are unchanged. Its corrected contributions define the spatial labels:

```text
e[t] = corrected contribution from the faults in interval t
D[t] = H e[t] XOR m[t] XOR m[t-1]
```

The syndrome and logical class of each contribution must equal those of the
physical frame difference. Directly labeling `F[t] XOR F[t-1]`, as v3 did, can
include stabilizer changes caused by older faults. Local weight reduction
does not always erase this history. The regression in
[`test_round_local_targets.py`](../../tests/test_round_local_targets.py) gives
identical late CNN inputs but different v3 targets, then verifies that v4
removes the conflict. This implements the independent-propagation principle in
[NVIDIA's Algorithm 3, Section III.3](https://arxiv.org/html/2607.10058v1#S3.SS3).

X checks detect Z errors and Z checks detect X errors. Full weight-4/8 face
stabilizers reduce spatial labels without changing syndrome or logical class.
Weight reduction precedes lexicographic tie fixing. Exact row-space membership
clears stabilizer-only targets; other cosets use a local heuristic, not an
exhaustive minimum-weight canonicalization.

For the prepared basis, the opening temporal reference follows the known
preparation. For the other basis, it is the first cycle's input-record parity
XOR the corrected input-frame syndrome. This uses that cycle's faults rather
than choosing a reference from future rounds. A single flipped record therefore
does not create a chain of labels in later rounds. The measured final temporal
boundary must independently close to zero.

A temporal head represents the interval between boundaries `t` and `t+1`.
It is supervised when either endpoint is observed, including the opening
interval of the unprepared basis. Masking that opening head would remove a
necessary correction. Temporal heads at the final boundary are excluded.
Spatial components commuting with preparation/readout are excluded at both
endpoints because those corrections affect neither detectors nor the logical
observable. Raw arrays retain them for inspection; loss and inference use the
mask.

Every generated batch verifies the actual masked model heads, not only the
unencoded labels. Their effects reconstruct every face detector and final
logical flip. Flag detectors remain unchanged and are passed to cleanup.
The local-target convention is still an adaptation; equivalence to all of
NVIDIA's delayed-fault and propagation protocols is not claimed.

## Compact tensors

Native data rows are two physical units apart. Compressing that spacing gives
a `d x d` grid, without quarter-unit centers enlarging it. Each face is assigned
to a distinct supporting data site near its center. Flags use distinct nearby
data anchors within each phase. Different channel types may share a site.
At distance 7, 31 data sites occupy a 7x7 grid.

The six inputs are X/Z events, X/Z geometric presence, flag events, and flag
presence. Face presence is weighted by face weight divided by 8, adapting the
paper's normalized-weight convention to this family. Flag presence is binary.
The four outputs are Z/X spatial differences and X/Z temporal corrections.
`target_mask` broadcasts over shots and excludes padding and undefined targets.

The pilot uses four 3x3x3 convolutions with widths 64,64,64,4, GeLU activation,
and dropout 0.01. Its spatial radius is 4; the worker verifies that it covers
the mapped face supports. Geometry coverage is necessary but does not prove
that every target can be inferred accurately from a local observation window.

## Generate and inspect

From the project root:

```bash
python -m training_data noah-training \
  --distance 3 --rounds 3 --basis Z --start-phase 0 --shots 32
```

NPZ exports contain inputs, labels, masks, physical boundary frames,
independently propagated spatial contributions, the opening reference, record
flips, spatial/temporal targets, and original detector and logical arrays.
JSON records schema, target protocol, anchors, source hashes, and conventions.
The NumPy/Stim notebook can inspect these without Torch.

## Train and evaluate in separate jobs

The pipeline is native 4.8.8 detectors, neural pre-decoder, residual detectors,
then Tesseract, the decoder behind every result in Noah's paper.

```bash
python scripts/submit_planar_pilot.py
```

The submission helper requests GPU training and a CPU evaluation job dependent
on successful training. GPU training requests one MIG GPU for 45 minutes, with
a 25-minute worker budget. Cleanup requests one CPU for 4 hours, with a
235-minute worker budget. Neither job depends on Della's overtime allowance.
QoS and CUDA device assignment remain under Slurm's control.

Training uses fresh batches and distinct calibration, validation, and test seed
ranges. Configuration rejects overlapping ranges. Checkpoints include weights,
optimizer, RNG states, schema, and embedding. Validation chooses the checkpoint
by masked loss and a correction threshold by face density subject to 95%
empirical target precision. Abstention is reported explicitly. This threshold
rule is a heuristic, not a guarantee about logical error rates.

For recipe studies, [`verification/tune_training_study.py`](../../verification/tune_training_study.py)
selects a correction threshold using paired cleanup-validation outcomes and
times, with abstention as a fallback. It reports deadline misses separately
from known logical failures. Pass its completed, checkpoint-bound report to
the evaluation command with `--policy PATH`. Reusing validation shots for
threshold selection does not make them an independent performance test.
The tuner accepts either shared `--thresholds` or repeated
`--threshold-pair SPATIAL TEMPORAL` probability cutoffs, including `1 1` for
abstention. Separate cutoffs follow NVIDIA's distinction between data and
syndrome correction thresholds. They change which predicted corrections are
applied; the detector-to-neural-to-residual-to-cleanup pipeline stays the same.

Every 4,000 steps also publishes the exact current model in
`milestones/step_004000/`, `milestones/step_008000/`, and so on. Each snapshot
contains `best.pt` and its own `report.json`, with a threshold selected from the
same validation data, per-head precision/recall, and residual event counts.
The snapshot is marked `fixed_training_step`; the parent run's `best.pt` still
selects by validation loss. The CPU evaluation script accepts a snapshot
directory as its training-output argument.

Training reports and checkpoints live in `verification/pilot_runs/JOBID/`.
CPU evaluation writes an `evaluation/` subdirectory. It compares cleanup alone
and pre-decoder plus cleanup on the same shots. It checks for stop signals
between paired shots, preserves completed prefixes, and can resume without
resampling or recounting completed shots. For an interrupted evaluation:

```bash
sbatch scripts/evaluate_planar_pilot.sbatch verification/pilot_runs/TRAIN_JOBID --resume
```

Use the same output directory, checkpoint, threshold, and sampling configuration
when resuming. Changing a batch size or seed does not create independent
continuation data. The decoder version and settings are part of the resume
identity. Low-confidence cleanup decodes count as failures; shots are never
silently discarded. Per-head precision/recall and
paired logical outcomes are reported, with per-case Wilson intervals.
New evaluation NPZ files also preserve each shot's cleanup durations for both
pipelines. Unknown historical timings remain NaN when an older prefix resumes.
Case reports separately measure CPU neural inference and correction conversion;
these are not GPU latency measurements. If the selected policy abstains, the
baseline result is reused, so zero extra cleanup time is not a speedup.
[`verification/analyze_paired_evaluation.py`](../../verification/analyze_paired_evaluation.py)
summarizes a completed evaluation with `--evaluation-output PATH`. It separates
face-event and all-detector reduction, includes CPU pre-decoder overhead, and
uses a paired bootstrap within each basis/phase to describe shot uncertainty.
Those intervals do not include hardware variation or model-selection effects.

Tesseract decodes the full native hypergraph DEM, flags included, with Noah's
published search from `data/decoder_configs.json`: beam 20 with climbing, 21
orderings from each of the BFS, coordinate, and index sources, seed 518278944,
and a 1,000,000 queue limit. Runtime depends strongly on the syndrome and CPU;
fewer detection events do not guarantee a faster residual search. The
installed wheel is newer than Noah's pinned revision, so generated orderings
need not be bit-identical to his runs. Residual probabilities are not learned
or reweighted. A small pilot cannot establish decoding improvement or
threshold behavior.

NVIDIA's paper uses Chromobius. This adaptation supports Tesseract and BP+LSD
for the native flagged detector model. Select BP+LSD with `--cleanup bp_lsd`
when evaluating or tuning. Compatibility with stock Chromobius is not provided.

For short CUDA and pipeline verification:

```bash
sbatch scripts/verify_planar_gpu.sbatch
```

That job runs CPU/GPU output/gradient checks, checkpoint reloads, and a
12-step fresh-data smoke run. It is not full training. CPU cleanup is submitted
separately using its completed training directory. Runtime setup, when needed:

```bash
bash scripts/setup_tesseract.sh
```

This installs only the pinned compatible decoder wheel under `.runtime/`.
It does not change the shared Torch environment or require manual `PYTHONPATH`.
