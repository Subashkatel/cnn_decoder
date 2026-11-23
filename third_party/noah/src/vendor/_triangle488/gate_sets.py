"""Declared native Clifford benchmarks, not claims about one universal chip."""
from dataclasses import dataclass

import stim


GATE_SET_CHOICES = ("cz", "cz_iswap", "cx_cxswap", "custom_clifford")
_PRESETS = {
    "cz": (("CZ",), "Canonical CZ-native benchmark on the declared grid."),
    "cz_iswap": (("CZ", "ISWAP", "ISWAP_DAG"),
        "Benchmark assumption: CZ, ISWAP and ISWAP_DAG are each calibrated native "
        "primitives on every declared edge, each taking one physical moment."),
    "cx_cxswap": (("CX", "CXSWAP", "SWAPCX"),
        "Benchmark assumption: CX, CXSWAP and SWAPCX are each calibrated native "
        "primitives on every declared edge and in either ordered orientation, "
        "each taking one physical moment. This is not a same-chip hardware claim."),
}


@dataclass(frozen=True)
class NativeGateSet:
    name: str
    two_qubit_gates: tuple[str, ...]
    hardware_assumptions: str

    def __post_init__(self):
        if self.name not in GATE_SET_CHOICES:
            raise ValueError(f"Unknown gate set: {self.name!r}")
        if not isinstance(self.hardware_assumptions, str) or not self.hardware_assumptions.strip():
            raise ValueError("Declare nonempty hardware assumptions for this gate set")
        if not isinstance(self.two_qubit_gates, tuple) or not self.two_qubit_gates:
            raise ValueError("Declare a nonempty tuple of native two-qubit gates")
        if len(set(self.two_qubit_gates)) != len(self.two_qubit_gates):
            raise ValueError("Native two-qubit gate names must be distinct")
        for name in self.two_qubit_gates:
            try:
                data = stim.gate_data(name)
            except (IndexError, TypeError, ValueError) as ex:
                raise ValueError(f"Unknown native Clifford gate: {name!r}") from ex
            if data.name != name or not data.is_unitary or not data.is_two_qubit_gate:
                raise ValueError(f"Use canonical Stim two-qubit unitary names: {name!r}")
        if self.name != "custom_clifford" and self.two_qubit_gates != _PRESETS[self.name][0]:
            raise ValueError("A named gate set cannot override its native primitives")

    @property
    def canonical_si1000(self):
        return self.name == "cz"

    def receipt(self):
        return dict(
            name=self.name, native_two_qubit_gates=list(self.two_qubit_gates),
            hardware_assumptions=self.hardware_assumptions,
            canonical_si1000=self.canonical_si1000,
            noise_model=("canonical native CZ SI1000" if self.canonical_si1000
                         else "SI1000-style gate-set extension (not canonical SI1000)"),
            two_qubit_noise="Independent DEPOLARIZE2(p) after each declared primitive",
            one_qubit_noise="Independent DEPOLARIZE1(p/10) after every physical one-qubit Clifford",
            reset_noise="X_ERROR(2p) after RZ",
            measurement_noise="X_ERROR(5p) before MZ; no post-measurement depolarization",
            idle_noise="Independent DEPOLARIZE1(p/10) per idle physical moment",
            collapse_wait_noise="Additional independent DEPOLARIZE1(2p) on every qubit not reset/measured in a collapse moment",
            primitive_cost="One physical moment per declared primitive; no free local-equivalence rotations",
            connectivity="Every two-qubit primitive uses a listed undirected grid edge; either target order is allowed",
            one_qubit_primitives="All canonical Stim unitary one-qubit Clifford gates",
            boundary_noise="No additional data-boundary depolarization",
            non_clifford_status="Not certified by this Stim adapter; independent extended checker required",
        )


def resolve_gate_set(gate_set="cz", native_two_qubit_gates=None, hardware_assumptions=None):
    """Resolve a preset or an explicit custom hardware benchmark declaration.

    Custom names must be canonical (e.g. CX, not CNOT), unitary and two-qubit.
    The declaration authorizes primitives; it does not synthesize a candidate.
    """
    if isinstance(gate_set, NativeGateSet):
        if native_two_qubit_gates is not None or hardware_assumptions is not None:
            raise ValueError("A resolved gate set cannot be combined with overrides")
        return gate_set
    if gate_set not in GATE_SET_CHOICES:
        raise ValueError(f"Unknown gate set: {gate_set!r}")
    if gate_set == "custom_clifford":
        if native_two_qubit_gates is None:
            raise ValueError("custom_clifford requires explicit native_two_qubit_gates")
        gates = (tuple(part.strip() for part in native_two_qubit_gates.split(","))
                 if isinstance(native_two_qubit_gates, str) else tuple(native_two_qubit_gates))
        return NativeGateSet(gate_set, gates, hardware_assumptions)
    if native_two_qubit_gates is not None:
        raise ValueError("Only custom_clifford accepts native_two_qubit_gates")
    gates, assumptions = _PRESETS[gate_set]
    return NativeGateSet(gate_set, gates,
                         assumptions if hardware_assumptions is None else hardware_assumptions)
