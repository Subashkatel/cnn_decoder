"""Scalable 24-moment native triangle rule by Bell-preparation polarity.

On the boundary Bell pairs that feed a routing ancilla in source slot1,
reverse the initial Bell CNOT and exchange the two preparation bases.
This prepares exactly the same Bell state. The second cycle is the complete
reverse, exchanging preparation and readout; its Bell records are relabeled
by the exact instrument calculation. Every cycle retains all flag records.

The native schedule preserves complete per-wire gate words after this change.
No per-size scheduling solve is used. Distance certificates are separate.
"""
import stim
from .native import compile_cx_cycle, split_moments
from .reference.cx_rule import build_cycle
from .reference.ideal_css import instrument

PERIOD = 2

def _inputs(geometry):
    forward = build_cycle(geometry, 0)
    routing = set(geometry['routing_ancillas'])
    feeds_routing = {a for a, b in forward['layers'][1] if b in routing}
    prepare = set(forward['h_prepare'])
    nd = len(geometry['qubit_coords'])
    for edge in forward['layers'][0]:
        a, b = edge
        if b in feeds_routing:
            assert a >= nd and b >= nd and a in prepare and b not in prepare
            edge[:] = [b, a]
            prepare ^= {a, b}
    forward['h_prepare'] = sorted(prepare)
    reverse = dict(forward, layers=list(reversed(forward['layers'])),
                   h_prepare=forward['h_measure'], h_measure=forward['h_prepare'])
    schedules = [forward, reverse]
    frames = []
    for schedule in schedules:
        last = {}
        for layer in schedule['layers']:
            for control, target in layer:
                if control < nd: last[control] = False
                if target < nd: last[target] = True
        frames.append(sorted(q for q, flag in last.items() if flag))
    return schedules, frames

def compact(circuit):
    layers, last, measurements = [], {}, []
    for moment in split_moments(circuit):
        for instruction in moment:
            targets = [t.value for t in instruction.targets_copy()]
            if instruction.name == 'M':
                measurements.extend(targets)
                continue
            stride = 2 if instruction.name == 'CZ' else 1
            for i in range(0, len(targets), stride):
                qs = targets[i:i+stride]
                depth = max((last.get(q, -1) for q in qs), default=-1) + 1
                while len(layers) <= depth: layers.append([])
                layers[depth].append((instruction.name, qs))
                for q in qs: last[q] = depth
    layers.append([('M', measurements)])
    out = stim.Circuit()
    for layer in layers:
        for gate, qs in layer: out.append(gate, qs)
        out.append('TICK')
    return out

def build_cycles(geometry):
    schedules, frames = _inputs(geometry)
    return [compact(compile_cx_cycle(s, frames[(p-1)%PERIOD], frames[p]))
            for p, s in enumerate(schedules)]

def build_maps(geometry):
    schedules, frames = _inputs(geometry)
    return [dict(instrument(geometry, s), input_h_frame=frames[(p-1)%PERIOD],
                 output_h_frame=frames[p]) for p, s in enumerate(schedules)]
