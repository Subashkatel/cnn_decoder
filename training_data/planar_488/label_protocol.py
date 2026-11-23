"""Version the correction-label convention independently of the physical samples."""

TRAJECTORY_PROTOCOL = "trajectory"
ROUND_LOCAL_PROTOCOL = "round_local"
DEFAULT_TARGET_PROTOCOL = ROUND_LOCAL_PROTOCOL
SCHEMA_BY_PROTOCOL = {
    ROUND_LOCAL_PROTOCOL: "planar_488_local_targets_v4",
    TRAJECTORY_PROTOCOL: "planar_488_local_targets_v3",
}


def schema_for_protocol(protocol):
    if protocol not in SCHEMA_BY_PROTOCOL:
        raise ValueError(f"Unknown correction-label protocol: {protocol}")
    return SCHEMA_BY_PROTOCOL[protocol]


def protocol_from_record(configuration):
    """Records without a protocol predate independent round-local labels."""
    return configuration.get("target_protocol", TRAJECTORY_PROTOCOL)


def schema_for_configuration(configuration):
    protocol = protocol_from_record(configuration)
    return schema_for_protocol(protocol)
