"""Build an explicitly framed X or Z memory from native complete cycles."""
import stim

from .gate_sets import resolve_gate_set
from .native import add_moment, validate_cycle
from .reference.binary import bits, indices


def build_memory(geometry, cycles, maps, basis="Z", rounds=3, start=0, *,
                 gate_set="cz", native_two_qubit_gates=None, hardware_assumptions=None):
    profile = resolve_gate_set(gate_set, native_two_qubit_gates, hardware_assumptions)
    if basis not in ("X", "Z") or type(rounds) is not int or rounds < 1:
        raise ValueError("Use X/Z and a positive integer number of complete cycles")
    if not cycles or len(cycles) != len(maps) or not 0 <= start < len(cycles):
        raise ValueError("Invalid phase period or starting phase")
    n = len(geometry["qubit_coords"])
    na = len(geometry["ancilla_coords"])
    for phase, (cycle, mapping) in enumerate(zip(cycles, maps)):
        validate_cycle(geometry, cycle, mapping, gate_set=profile)
        following = maps[(phase+1) % len(maps)]
        if set(mapping.get("output_h_frame", [])) != set(following.get("input_h_frame", [])):
            raise ValueError("Boundary frames do not compose across the period")
        if mapping["logical_support"] != following["logical_support"]:
            raise ValueError("Declare a consistent logical representative across phases")
    c = stim.Circuit()
    for q, coord in enumerate(geometry["qubit_coords"] + geometry["ancilla_coords"]):
        c.append("QUBIT_COORDS", [q], coord)
    add_moment(c, "R", range(n))
    initial_h = set(range(n)) if basis == "X" else set()
    initial_h ^= set(maps[start].get("input_h_frame", []))
    add_moment(c, "H", sorted(initial_h))
    nm = 0

    def declare(name, mask, args):
        c.append(name, [stim.target_rec(i-nm) for i in indices(mask)], args)

    for rd in range(rounds):
        phase = (start+rd) % len(cycles)
        mapping = maps[phase]
        previous = maps[(phase-1) % len(maps)]
        offset = nm
        c += cycles[phase]
        nm += na
        for j, mask in enumerate(mapping["record_constraints"]):
            declare("DETECTOR", mask << offset, [j, 1, rd])
        for fi in range(len(geometry["plaquettes"])):
            axes = basis if rd == 0 else "XZ"
            for axis in axes:
                mask = mapping["input_face_record_maps"][axis][fi] << offset
                if rd:
                    mask ^= previous["output_face_record_maps"][axis][fi] << (offset-na)
                declare("DETECTOR", mask, [fi, 0, rd, 0 if axis == "X" else 3])
        declare("OBSERVABLE_INCLUDE", mapping["logical_choi_signs"][basis][0] << offset, [0])
    final_h = set(range(n)) if basis == "X" else set()
    final_h ^= set(mapping.get("output_h_frame", []))
    add_moment(c, "H", sorted(final_h))
    data_offset = nm
    add_moment(c, "M", range(n))
    nm += n
    for fi, face in enumerate(geometry["plaquettes"]):
        mask = (mapping["output_face_record_maps"][basis][fi] << offset) ^ bits(data_offset+q for q in face["qubits"])
        declare("DETECTOR", mask, [fi, 0, rounds, 0 if basis == "X" else 3])
    declare("OBSERVABLE_INCLUDE", bits(data_offset+q for q in mapping["logical_support"]), [0])
    return c
