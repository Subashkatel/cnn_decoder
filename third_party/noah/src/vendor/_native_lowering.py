from __future__ import annotations
from typing import Sequence
import stim

_ANNOTATIONS = {
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
}

_PURE_NOISE = {
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "E",
    "ELSE_CORRELATED_ERROR",
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
}

_COLLAPSE = {"R", "RX", "M", "MX"}

_SUPPORTED_IDEAL = _COLLAPSE | {"H", "CX", "CZ", "SWAP"}

class Baseline488FreezerError(ValueError):
    """The proposed baseline artifact does not satisfy the frozen contract."""

def _qubit_targets(instruction: stim.CircuitInstruction) -> tuple[int, ...]:
    targets = instruction.targets_copy()
    if any(not target.is_qubit_target for target in targets):
        raise Baseline488FreezerError(
            f"{instruction.name} has a non-qubit target in the ideal schedule"
        )
    return tuple(target.qubit_value for target in targets)

def _split_source_moments(
    circuit: stim.Circuit,
) -> tuple[tuple[stim.CircuitInstruction, ...], ...]:
    """Flatten and split the clean memory at every physical-source TICK.

    Bell compilers may place their final detector/observable table after a
    presentation-only trailing TICK.  Such a terminal annotation tail is
    folded onto the final readout moment: it is not a physical idle moment and
    therefore must not acquire SI1000 idle noise.  Annotation-only moments in
    the interior remain invalid.
    """

    moments: list[tuple[stim.CircuitInstruction, ...]] = []
    current: list[stim.CircuitInstruction] = []
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            if not current:
                raise Baseline488FreezerError(
                    "clean memory contains an empty TICK moment"
                )
            moments.append(tuple(current))
            current.clear()
        else:
            current.append(instruction)
    if current:
        moments.append(tuple(current))
    if not moments:
        raise Baseline488FreezerError("clean memory contains no source moments")
    while len(moments) > 1 and all(
        instruction.name in _ANNOTATIONS for instruction in moments[-1]
    ):
        moments[-2] = (*moments[-2], *moments[-1])
        moments.pop()
    return tuple(moments)

def _split_annotations(
    moment: Sequence[stim.CircuitInstruction],
) -> tuple[
    tuple[stim.CircuitInstruction, ...],
    tuple[stim.CircuitInstruction, ...],
    tuple[stim.CircuitInstruction, ...],
]:
    """Return prefix annotations, ideal quantum ops, and suffix annotations."""

    first_quantum = None
    last_quantum = None
    for index, instruction in enumerate(moment):
        name = instruction.name
        if name in _PURE_NOISE or (
            name in {"M", "MX"} and instruction.gate_args_copy()
        ):
            raise Baseline488FreezerError(
                "clean baseline contains a stochastic instruction before freezing"
            )
        if name in _ANNOTATIONS:
            continue
        if name not in _SUPPORTED_IDEAL:
            if name == "CXSWAP":
                raise Baseline488FreezerError(
                    "CXSWAP is forbidden by the 4.8.8 baseline model"
                )
            raise Baseline488FreezerError(
                f"unsupported ideal operation {name!r} in baseline memory"
            )
        if any(
            target.is_measurement_record_target for target in instruction.targets_copy()
        ):
            raise Baseline488FreezerError(
                f"record-controlled quantum operation {name!r} is forbidden"
            )
        first_quantum = index if first_quantum is None else first_quantum
        last_quantum = index
    if first_quantum is None or last_quantum is None:
        raise Baseline488FreezerError(
            "every flattened source moment must contain a physical operation"
        )
    prefix = tuple(moment[:first_quantum])
    quantum = tuple(moment[first_quantum : last_quantum + 1])
    suffix = tuple(moment[last_quantum + 1 :])
    if any(instruction.name not in _ANNOTATIONS for instruction in (*prefix, *suffix)):
        raise Baseline488FreezerError(
            "clean annotations are interleaved with physical operations"
        )
    if any(instruction.name in _ANNOTATIONS for instruction in quantum):
        raise Baseline488FreezerError(
            "clean annotations are interleaved with physical operations"
        )
    return prefix, quantum, suffix

def _lower_quantum_moment(
    quantum: Sequence[stim.CircuitInstruction],
    *,
    preserve_mx_targets: Sequence[int] = (),
) -> tuple[tuple[stim.CircuitInstruction, ...], ...]:
    """Lower RX/MX into Stimflow-supported R/H/M physical moments."""

    names = {instruction.name for instruction in quantum}
    if names & _COLLAPSE:
        if not names <= _COLLAPSE:
            raise Baseline488FreezerError(
                "collapse and unitary operations share one source moment"
            )
        measurement = [
            instruction for instruction in quantum if instruction.name in {"M", "MX"}
        ]
        reset = [
            instruction for instruction in quantum if instruction.name in {"R", "RX"}
        ]
        mx_targets = {
            target
            for instruction in measurement
            if instruction.name == "MX"
            for target in _qubit_targets(instruction)
        }
        if not set(preserve_mx_targets) <= mx_targets:
            raise Baseline488FreezerError(
                "post-measurement H requested for a target not measured by MX"
            )
        pre = stim.Circuit()
        for instruction in measurement:
            if instruction.name == "MX":
                pre.append("H", _qubit_targets(instruction))
        collapse = stim.Circuit()
        # Measurements precede resets exactly as in the repository's pinned
        # Stimflow physical-collapse lowering.  Resets do not consume records.
        for instruction in measurement:
            collapse.append("M", _qubit_targets(instruction))
        for instruction in reset:
            collapse.append("R", _qubit_targets(instruction))
        post = stim.Circuit()
        post_h_targets: set[int] = set(preserve_mx_targets)
        for instruction in reset:
            if instruction.name == "RX":
                post_h_targets.update(_qubit_targets(instruction))
        if post_h_targets:
            post.append("H", sorted(post_h_targets))
        return tuple(tuple(part) for part in (pre, collapse, post) if len(part))
    return (tuple(quantum),)

def _mx_post_h_targets_by_source_moment(
    source_moments: Sequence[Sequence[stim.CircuitInstruction]],
) -> dict[int, tuple[int, ...]]:
    """Find MX projections whose post-measurement state is later consumed.

    ``H;M`` reproduces an X-basis measurement bit but leaves a Z eigenstate.
    When that wire is used before reset, an additional H is required to
    reproduce the ideal MX instrument.  The scan is deliberately on physical
    wires: a SWAP itself is a consuming operation, so the state is restored
    before movement and then follows the actual wire exchange.
    """

    quantum_by_moment = tuple(
        _split_annotations(moment)[1] for moment in source_moments
    )
    result: dict[int, tuple[int, ...]] = {}
    for source_index, quantum in enumerate(quantum_by_moment):
        measured = {
            target
            for instruction in quantum
            if instruction.name == "MX"
            for target in _qubit_targets(instruction)
        }
        live = []
        for target in sorted(measured):
            for later in quantum_by_moment[source_index + 1 :]:
                reset_here = any(
                    instruction.name in {"R", "RX"}
                    and target in _qubit_targets(instruction)
                    for instruction in later
                )
                if reset_here:
                    break
                consumed_here = any(
                    instruction.name in {"H", "CX", "CZ", "SWAP", "M", "MX"}
                    and target in _qubit_targets(instruction)
                    for instruction in later
                )
                if consumed_here:
                    live.append(target)
                    break
        if live:
            result[source_index] = tuple(live)
    return result
def lower_reference(clean):
    """Physical RX/MX conversion, retaining the pinned CZ compilation's moments."""
    if clean != clean.without_noise():
        raise ValueError("reference lowering requires noiseless input")
    moments = _split_source_moments(clean)
    if _mx_post_h_targets_by_source_moment(moments):
        raise ValueError("reference MX output is consumed before reset")
    out = stim.Circuit()
    for moment in moments:
        prefix, quantum, suffix = _split_annotations(moment)
        lowered = _lower_quantum_moment(quantum)
        for i, physical in enumerate(lowered):
            for inst in prefix if i == 0 else ():
                out.append(inst)
            for inst in physical:
                if inst.name not in {"R", "H", "CZ", "M"}:
                    raise ValueError(f"unsupported native reference operation: {inst.name}")
                out.append(inst)
            for inst in suffix if i == len(lowered)-1 else ():
                out.append(inst)
            out.append("TICK")
    if (out.num_measurements, out.num_detectors, out.num_observables) != (clean.num_measurements, clean.num_detectors, clean.num_observables):
        raise ValueError("reference lowering changed the record inventory")
    return out
