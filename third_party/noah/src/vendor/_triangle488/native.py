"""Declared native Clifford scheduling, SI1000-style noise and ideal contract.

No distance solver, fault search, decoder, or Monte Carlo is invoked here.
"""
from collections import Counter
import stim

from .gate_sets import resolve_gate_set
from .reference.binary import bits, indices, rank

ANNOTATIONS = {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}
ONE_QUBIT = {data.name for data in stim.gate_data().values()
             if data.is_unitary and data.is_single_qubit_gate}
# Backward-compatible default alphabet; validators resolve the selected profile.
PHYSICAL = {"R", "CZ", "M"} | ONE_QUBIT


def add_moment(circuit, gate, targets):
    targets = list(targets)
    if targets:
        circuit.append(gate, targets)
        circuit.append("TICK")


def split_moments(circuit):
    result, current = [], []
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            # Empty TICK intervals are physical all-idle moments. A trailing
            # TICK ends the preceding moment without adding one at EOF.
            result.append(current)
            current = []
        else:
            current.append(instruction)
    if current:
        if any(i.name not in ANNOTATIONS for i in current):
            result.append(current)
        elif result:
            result[-1].extend(current)
    return result


def compile_cx_cycle(schedule, input_h_frame=(), output_h_frame=()):
    """CX(c,t)=H(t) CZ(c,t) H(t), cancelling commuting target H gates.

    All preparation and measurement rotations are physical. This compiler
    respects the complete-cycle barriers and uses no cross-cycle entanglers.
    """
    c = stim.Circuit()
    add_moment(c, "R", schedule["measurement_order"])
    pending = set(schedule["h_prepare"]) ^ set(input_h_frame)
    for layer in schedule["layers"]:
        targets = {t for _, t in layer}
        active = {q for pair in layer for q in pair}
        add_moment(c, "H", sorted((pending ^ targets) & active))
        add_moment(c, "CZ", [q for pair in layer for q in pair])
        pending = (pending - active) | targets
    add_moment(c, "H", sorted(pending ^ set(schedule["h_measure"]) ^ set(output_h_frame)))
    add_moment(c, "M", schedule["measurement_order"])
    return c


def cx_reference_cycle(schedule):
    c = stim.Circuit()
    add_moment(c, "R", schedule["measurement_order"])
    add_moment(c, "H", schedule["h_prepare"])
    for layer in schedule["layers"]:
        add_moment(c, "CX", [q for pair in layer for q in pair])
    add_moment(c, "H", schedule["h_measure"])
    add_moment(c, "M", schedule["measurement_order"])
    return c


def census(circuit, qubits, *, gate_set="cz", native_two_qubit_gates=None,
           hardware_assumptions=None):
    profile = resolve_gate_set(gate_set, native_two_qubit_gates, hardware_assumptions)
    two_qubit = set(profile.two_qubit_gates)
    physical = {"R", "M"} | ONE_QUBIT | two_qubit
    counts = Counter()
    gates = Counter()
    for moment in split_moments(circuit):
        counts["physical_moments"] += 1
        used = set()
        names = set()
        collapsing = set()
        for instruction in moment:
            name = instruction.name
            if name in ANNOTATIONS:
                continue
            if name not in physical:
                raise ValueError(f"Non-native operation: {instruction}")
            if instruction.gate_args_copy():
                raise ValueError("Physical input must be a noiseless Clifford circuit")
            targets = instruction.targets_copy()
            if not all(t.is_qubit_target for t in targets):
                raise ValueError("Only physical qubit targets are allowed")
            qs = [t.value for t in targets]
            if len(set(qs)) != len(qs) or used.intersection(qs):
                raise ValueError("A qubit has multiple operations in one physical moment")
            if not set(qs) <= set(range(qubits)):
                raise ValueError("Operation leaves the hardware")
            used.update(qs)
            names.add(name)
            if name in {"R", "M"}:
                collapsing.update(qs)
            gates[name] += len(qs) // (2 if name in two_qubit else 1)
        idle = qubits - len(used)
        counts["idle_qubit_moments"] += idle
        counts["cz_layers"] += "CZ" in names
        counts["two_qubit_layers"] += bool(names & two_qubit)
        counts["h_layers"] += "H" in names
        counts["one_qubit_clifford_layers"] += bool(names & ONE_QUBIT)
        if names & {"R", "M"}:
            counts["collapse_moments"] += 1
            counts["resonator_wait_qubit_moments"] += qubits-len(collapsing)
    return dict(counts, gates=dict(sorted(gates.items())), gate_set=profile.receipt())


def decorate_si1000(clean, qubits, p, *, gate_set="cz", native_two_qubit_gates=None,
                    hardware_assumptions=None):
    """Canonical CZ SI1000 by default; declared other sets are extensions.

    R -> X_ERROR(2p); one-qubit Clifford -> D1(p/10); native 2q -> D2(p);
    M -> X_ERROR(5p) before noiseless M, with no post-measurement D1.
    Every idle -> D1(p/10). In a moment containing R or M, every qubit
    not reset/measured receives D1(2p), including concurrently active gates.
    """
    if not 0 <= p <= 0.2:
        raise ValueError("p must be in [0, 0.2]")
    profile = resolve_gate_set(gate_set, native_two_qubit_gates, hardware_assumptions)
    census(clean, qubits, gate_set=profile)
    out = stim.Circuit()
    for moment in split_moments(clean):
        active, collapsing = set(), set()
        for instruction in moment:
            if instruction.name in ANNOTATIONS:
                out.append(instruction)
                continue
            name = instruction.name
            qs = [t.value for t in instruction.targets_copy()]
            active.update(qs)
            if name in {"R", "M"}:
                collapsing.update(qs)
            if name == "M":
                if p:
                    out.append("X_ERROR", qs, 5*p)
                out.append("M", qs)
            else:
                out.append(instruction)
                if p:
                    noise, probability = (("DEPOLARIZE1", p/10) if name in ONE_QUBIT
                        else ("X_ERROR", 2*p) if name == "R" else ("DEPOLARIZE2", p))
                    out.append(noise, qs, probability)
        idle = sorted(set(range(qubits)) - active)
        if p and idle:
            out.append("DEPOLARIZE1", idle, p/10)
        waiting = sorted(set(range(qubits)) - collapsing)
        if p and collapsing and waiting:
            out.append("DEPOLARIZE1", waiting, 2*p)
        out.append("TICK")
    return out


def _pauli(qubits, axis, support, h_frame=()):
    result = stim.PauliString(qubits)
    h_frame = set(h_frame)
    for q in support:
        result[q] = ({"X": "Z", "Z": "X"}[axis] if q in h_frame else axis)
    return result


def validate_cycle(geometry, cycle, maps, *, gate_set="cz", native_two_qubit_gates=None,
                   hardware_assumptions=None):
    """Check hardware and the intended signed extraction instrument directly.

    This is stronger than checking deterministic memory detectors: input and
    output face observables AND both logical identities must have signed flows.
    Candidate record maps may differ from the starter's maps.
    """
    n = len(geometry["qubit_coords"])
    q = n + len(geometry["ancilla_coords"])
    ancillas = set(range(n, q))
    edges = {tuple(sorted(e)) for e in geometry["gate_edges"]}
    profile = resolve_gate_set(gate_set, native_two_qubit_gates, hardware_assumptions)
    metrics = census(cycle, q, gate_set=profile)
    reset, measured, measurement_order = set(), set(), []
    for moment in split_moments(cycle):
        for instruction in moment:
            name = instruction.name
            if name in ANNOTATIONS:
                raise ValueError("Cycle annotations belong in the declared record maps")
            qs = [t.value for t in instruction.targets_copy()]
            if instruction.gate_args_copy():
                raise ValueError("Candidate cycles must be noiseless")
            if name == "R":
                if not set(qs) <= ancillas or reset.intersection(qs):
                    raise ValueError("Each ancilla resets exactly once; data never resets")
                reset.update(qs)
            else:
                if (set(qs) & ancillas) - reset or set(qs) & measured:
                    raise ValueError("Ancilla operations must be between reset and measurement")
                if name == "M":
                    if not set(qs) <= ancillas:
                        raise ValueError("Data cannot be measured during extraction")
                    measured.update(qs)
                    measurement_order.extend(qs)
                if name in profile.two_qubit_gates:
                    for a, b in zip(qs[::2], qs[1::2]):
                        if tuple(sorted((a,b))) not in edges:
                            raise ValueError(f"{name} must use a listed nearest-neighbor grid edge")
    if reset != ancillas or measured != ancillas:
        raise ValueError("A full cycle must reset and measure every ancilla once")
    na = len(ancillas)
    faces = [f["qubits"] for f in geometry["plaquettes"]]
    fbits = [bits(f) for f in faces]
    if rank(fbits) != (n-1)//2 or any((a&b).bit_count()%2 for a in fbits for b in fbits):
        raise ValueError("Geometry is not the intended one-logical-qubit CSS code")
    logical = maps["logical_support"]
    lo = bits(logical)
    if not set(logical) <= set(range(n)) or not lo.bit_count()%2 or any((lo&f).bit_count()%2 for f in fbits):
        raise ValueError("Invalid logical X/Z representative")
    constraints = maps["record_constraints"]
    signed_flows = []
    zero = stim.PauliString(q)
    all_masks = list(constraints)
    frames = {side: maps.get(f"{side}_h_frame", []) for side in ("input", "output")}
    if any(len(f) != len(set(f)) or not set(f) <= set(range(n)) for f in frames.values()):
        raise ValueError("Boundary H frames must be distinct physical data-qubit indices")
    for axis in "XZ":
        for side in ("input", "output"):
            masks = maps[f"{side}_face_record_maps"][axis]
            if len(masks) != len(faces):
                raise ValueError("Every face requires a record map in each basis")
            all_masks.extend(masks)
            for face, mask in zip(faces, masks):
                pauli = _pauli(q, axis, face, frames[side])
                signed_flows.append(stim.Flow(input=pauli if side=="input" else zero,
                    output=pauli if side=="output" else zero, measurements=list(indices(mask))))
        masks = maps["logical_choi_signs"][axis]
        if len(masks) != 1:
            raise ValueError("One logical identity map is required per basis")
        all_masks.extend(masks)
        signed_flows.append(stim.Flow(input=_pauli(q, axis, logical, frames["input"]),
            output=_pauli(q, axis, logical, frames["output"]), measurements=list(indices(masks[0]))))
    if any(type(m) is not int or m < 0 or m >> na for m in all_masks):
        raise ValueError("Record map outside the cycle's measurements")
    for mask in constraints:
        signed_flows.append(stim.Flow(input=zero, output=zero, measurements=list(indices(mask))))
    syndrome = [m for axis in "XZ" for m in maps["input_face_record_maps"][axis]]
    if rank(syndrome + constraints) != 2*len(faces) + rank(constraints):
        raise ValueError("Both face syndrome sets must be independently recoverable")
    if not cycle.has_all_flows(signed_flows, unsigned=False):
        raise ValueError("The intended signed syndrome/logical instrument failed")
    return dict(status="PASS_IDEAL_NATIVE_INSTRUMENT", measurement_order=measurement_order,
                signed_flows_checked=len(signed_flows), **metrics)
