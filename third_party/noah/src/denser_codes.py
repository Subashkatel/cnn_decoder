"""Portable circuit construction for the memories studied in the accompanying paper.

No sampling, decoding, distance searches, network access or repository access.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import stim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENDOR = HERE / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from _triangle488 import geometry as triangle_geometry
from _triangle488 import candidate, memory as triangle_memory, native
from _triangle488.reference import cx_rule, ideal_css
from _native_lowering import lower_reference
from _coordinates import decoder_coordinates

FAMILIES = ("planar_surface", "planar_666", "planar_488", "toric",
            "torus_666_best", "torus_666_alternate", "torus_488")


@lru_cache(maxsize=32)
def _triangle_components(distance, gate_set):
    g = triangle_geometry.generate_base_layout((distance + 1) // 4)
    if gate_set == "native-cz":
        return g, candidate.build_cycles(g), candidate.build_maps(g)
    schedules = [cx_rule.build_cycle(g, phase) for phase in range(2)]
    cycles = [native.cx_reference_cycle(s) for s in schedules]
    return g, cycles, [ideal_css.instrument(g, s) for s in schedules]


def _planar_baseline(family, distance, basis, rounds, gate_set):
    import gen
    if family == "planar_surface":
        from clorco.surface_code._xz_surface_code_memory_circuits import make_xz_memory_experiment_chunks
        c = gen.compile_chunks_into_circuit(make_xz_memory_experiment_chunks(
            basis=basis, diameter=distance, rounds=rounds))
    else:
        from clorco.color_code._superdense_planar_color_code_circuits import make_superdense_color_code_circuit
        c = make_superdense_color_code_circuit(base_data_width=distance, basis=basis, rounds=rounds)
    if gate_set == "native-cz":
        c = lower_reference(gen.transpile_to_z_basis_interaction_circuit(c))
    return c


def _torus_baseline(family, distance, basis, rounds, gate_set):
    """Construct a periodic family, retaining exact historical d=4,8 recipes."""
    import gen
    if distance in (4, 8):
        path = ROOT / "construction_templates" / f"{family}_d{distance}_{basis}.stim"
        c = stim.Circuit(path.read_text())
    else:
        from _baseline_torus import build_memory as build
        c = build(family, distance, basis, rounds)
    if distance in (4, 8) and rounds != distance:
        out, repeats = stim.Circuit(), 0
        for inst in c:
            if isinstance(inst, stim.CircuitRepeatBlock):
                if inst.repeat_count != distance - 1:
                    raise ValueError("Unexpected reference repeat structure")
                repeats += 1
                if rounds > 1:
                    out.append(stim.CircuitRepeatBlock(rounds - 1, inst.body_copy()))
            else:
                out.append(inst)
        if repeats != 1:
            raise ValueError("Expected one repeated extraction block")
        c = out
    if gate_set == "native-cz":
        c = lower_reference(gen.transpile_to_z_basis_interaction_circuit(c, is_entire_circuit=True))
    return c


@lru_cache(maxsize=32)
def _torus_native_components(distance):
    from _torus488.family import make_cycle
    from _torus488.geometry import build_geometry
    from _torus488.check import validate
    g = build_geometry(distance)
    return g, validate(make_cycle(distance), g)


def _torus488(distance, basis, rounds, gate_set):
    if gate_set == "native-cz":
        from _torus488.memory25 import build_memory as build
        from _torus488.check import cb
        g, checked = _torus_native_components(distance)
        # The archived constructor also verifies the full signed instrument.
        # Its historical decorator is not used: common canonical SI1000 below
        # supplies the exact simulation convention for all seven families.
        result = build(distance, basis, rounds, .001, geometry=g, checked=checked)
        return stim.Circuit(cb(result["clean"]).decode()).without_noise()
    from _torus488_cnot.geometry import build_geometry
    from _torus488_cnot.rule import build_schedule
    from _torus488_cnot.instrument import instrument
    from _torus488_cnot.make_memory import memory
    g, s = build_geometry(distance), build_schedule(distance)
    a = instrument(s, [f["data_support"] for f in g["plaquettes"]], g["canonical_logicals"])
    # The historical emitter coalesces equal-probability channels; retain its
    # nonzero bookkeeping probabilities and discard all noise immediately.
    c, _ = memory(g, [s] * rounds, basis, p=.001, checked=[a] * rounds)
    return stim.Circuit("\n".join(map(str, c))).without_noise()


def build_memory(family, distance, basis="Z", *, rounds=None, start_phase=0,
                 gate_set="native-cz", p=0):
    """Return a Stim memory circuit in the paper's detector-coordinate convention.

    ``p=0`` gives an ideal circuit. Positive p applies canonical SI1000 and
    requires native-cz. The CNOT representations are explanatory constructions,
    not a different noise model, and need not be bytewise native-transpilations
    of the adopted optimized 4.8.8 circuits.
    """
    if family not in FAMILIES or basis not in ("X", "Z"):
        raise ValueError("Use a listed family and memory basis X or Z")
    if type(distance) is not int or distance < 3:
        raise ValueError("distance must be an integer >= 3")
    if gate_set not in ("cnot", "native-cz"):
        raise ValueError("gate_set must be cnot or native-cz")
    if not 0 <= p <= .2 or (p and gate_set != "native-cz"):
        raise ValueError("SI1000 requires native-cz and 0 <= p <= .2")
    if family.startswith("planar_") and distance % 2 != 1:
        raise ValueError("Planar memories require odd distance")
    if family == "planar_488" and distance % 4 != 3:
        raise ValueError("The planar 4.8.8 family has d=4m-1")
    if not family.startswith("planar_") and distance % 4:
        raise ValueError("The included torus families have d=4m")
    if type(start_phase) is not int or start_phase not in ((0, 1) if family == "planar_488" else (0,)):
        raise ValueError("Only planar 4.8.8 has two starting phases")
    rounds = distance if rounds is None else rounds
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if family == "planar_666" and rounds < 2:
        raise ValueError("The pinned planar 6.6.6 constructor requires at least two rounds")
    geometry = None
    if family == "planar_488":
        geometry, cycles, maps = _triangle_components(distance, gate_set)
        options = {} if gate_set == "native-cz" else dict(gate_set="custom_clifford",
            native_two_qubit_gates=("CX",), hardware_assumptions="Ideal CNOT representation only")
        c = triangle_memory.build_memory(geometry, cycles, maps, basis, rounds, start_phase, **options)
    elif family in ("planar_surface", "planar_666"):
        c = _planar_baseline(family, distance, basis, rounds, gate_set)
    elif family == "torus_488":
        c = _torus488(distance, basis, rounds, gate_set)
    else:
        c = _torus_baseline(family, distance, basis, rounds, gate_set)
    if gate_set == "native-cz":
        c = native.decorate_si1000(c, c.num_qubits, p)
        c, _ = decoder_coordinates(c, family=family, geometry=geometry)
    return c


def circuit_bytes(circuit):
    return (str(circuit).rstrip() + "\n").encode()


def build_decoder_dem(circuit):
    """Return the same full hypergraph DEM convention supplied to Tesseract.

    Disjoint physical-channel branches are approximated as independent errors
    by Stim. Hyperedges are retained; there is no graphlike decomposition or
    sparsification. Last-bit float formatting can vary across CPU platforms.
    """
    return circuit.detector_error_model(approximate_disjoint_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--basis", choices=("X", "Z"), default="Z")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--gate-set", choices=("cnot", "native-cz"), default="native-cz")
    parser.add_argument("--p", type=float, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dem-out", type=Path, help="Optionally export the Tesseract-input detector error model")
    args = parser.parse_args()
    circuit = build_memory(args.family, args.distance, args.basis, rounds=args.rounds,
        start_phase=args.start_phase, gate_set=args.gate_set, p=args.p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(circuit_bytes(circuit))
    if args.dem_out is not None:
        args.dem_out.parent.mkdir(parents=True, exist_ok=True)
        args.dem_out.write_bytes(circuit_bytes(build_decoder_dem(circuit)))
    print(json.dumps(dict(qubits=circuit.num_qubits, measurements=circuit.num_measurements,
                         detectors=circuit.num_detectors, observables=circuit.num_observables)))


if __name__ == "__main__":
    main()
