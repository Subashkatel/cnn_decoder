import stim
ContractError = ValueError

def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        raise ContractError("cannot assign a spatial centroid to empty support")
    return tuple(sum(p[axis] for p in points) / len(points) for axis in (0, 1))

def decoder_coordinates(circuit: stim.Circuit, *, family: str,
                        geometry: dict | None = None) -> tuple[stim.Circuit, dict]:
    """Change only detector metadata to spatial x,y,t; retain the fault graph."""
    flat = circuit.flattened()
    coordinates = flat.get_final_qubit_coordinates()
    detector_coords = flat.get_detector_coordinates()
    measured: list[int] = []
    result = stim.Circuit()
    rows = []
    face_centers = []
    if family == "planar_488":
        if geometry is None:
            raise ContractError("planar face geometry required")
        for face in geometry["plaquettes"]:
            face_centers.append(_centroid([tuple(geometry["qubit_coords"][q])
                                           for q in face["qubits"]]))
    for instruction in flat:
        if instruction.name == "M":
            measured.extend(t.value for t in instruction.targets_copy())
        if instruction.name != "DETECTOR":
            result.append(instruction)
            continue
        index = len(rows)
        old = detector_coords[index]
        if len(old) < 3:
            raise ContractError(f"detector {index} lacks a spacetime coordinate")
        method = "existing_spatial_coordinate"
        if family == "planar_488" and len(old) == 4 and old[1] == 0:
            fi = int(old[0])
            if old[0] != fi or not 0 <= fi < len(face_centers):
                raise ContractError("invalid planar face index")
            x, y = face_centers[fi]
            method = "data_face_centroid"
        elif (family == "planar_488" or (family.startswith("torus_") and old[:2] == [0, 0])):
            support = set()
            for target in instruction.targets_copy():
                if not target.is_measurement_record_target:
                    raise ContractError("non-record detector target")
                record = len(measured) + target.value
                if not 0 <= record < len(measured):
                    raise ContractError("invalid detector measurement record")
                support.add(measured[record])
            x, y = _centroid([tuple(coordinates[q][:2]) for q in sorted(support)])
            method = "measured_support_centroid"
        else:
            x, y = old[:2]
        new = (x, y, old[2])
        result.append("DETECTOR", instruction.targets_copy(), new, tag=instruction.tag)
        rows.append(dict(detector=index, original=old, xyz=new, method=method))
    # flattened() resolves all SHIFT_COORDS, so absolute detector arguments are safe.
    if len(rows) != circuit.num_detectors:
        raise ContractError("coordinate rewrite changed detector count")
    return result, {"dimensions": ["x", "y", "t"], "color_axis_removed": True,
                    "fault_graph_unchanged": True, "detectors": rows}
