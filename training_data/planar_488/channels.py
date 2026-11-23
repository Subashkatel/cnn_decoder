"""Named tensor channels and the X/Z axis order shared by detectors, frames, and tensors."""

from enum import IntEnum


# Face arrays order check bases X then Z; frames order error components X then Z.
PAULI_ORDER = ("X", "Z")


def check_axis(check_basis: str) -> int:
    """Position of a check basis on the X/Z axis of face arrays."""
    return PAULI_ORDER.index(check_basis)


def error_component(pauli: str) -> int:
    """Position of a Pauli error type on the X/Z axis of frames and spatial targets."""
    return PAULI_ORDER.index(pauli)


def flipping_component(basis: str) -> int:
    """Errors of the other Pauli type are the ones that flip a measurement in this basis."""
    same_component = error_component(basis)
    return 1 - same_component


class InputChannel(IntEnum):
    X_DETECTION_EVENT = 0
    Z_DETECTION_EVENT = 1
    X_PRESENT = 2
    Z_PRESENT = 3
    FLAG_EVENT = 4
    FLAG_PRESENT = 5


class TargetChannel(IntEnum):
    Z_ERROR_DIFFERENCE = 0
    X_ERROR_DIFFERENCE = 1
    X_TEMPORAL_RESIDUAL = 2
    Z_TEMPORAL_RESIDUAL = 3


def channel_names(channels: type[IntEnum]) -> list[str]:
    """Exported metadata names are the lowercase member names, in channel order."""
    names = []
    for channel in channels:
        lowercase_name = channel.name.lower()
        names.append(lowercase_name)
    return names


INPUT_CHANNELS = channel_names(InputChannel)
TARGET_CHANNELS = channel_names(TargetChannel)

# Rows are (check axis, event input, presence input, temporal target) for each face basis.
FACE_CHANNELS = (
    (
        check_axis("X"),
        InputChannel.X_DETECTION_EVENT,
        InputChannel.X_PRESENT,
        TargetChannel.X_TEMPORAL_RESIDUAL,
    ),
    (
        check_axis("Z"),
        InputChannel.Z_DETECTION_EVENT,
        InputChannel.Z_PRESENT,
        TargetChannel.Z_TEMPORAL_RESIDUAL,
    ),
)
# Rows are (error component, spatial target) for each Pauli error type.
SPATIAL_CHANNELS = (
    (error_component("X"), TargetChannel.X_ERROR_DIFFERENCE),
    (error_component("Z"), TargetChannel.Z_ERROR_DIFFERENCE),
)
