"""The Tesseract cleanup decoder, configured with Noah's published search settings."""

from dataclasses import dataclass
import importlib.metadata

import numpy as np

from training_data.planar_488.dependencies import load_tesseract


# Noah's published search, recorded in
# denser_planar_color_codes_source_v1/data/decoder_configs.json.
# Lists keep this record identical after a JSON round trip, which evaluation resumes compare.
PUBLISHED_CONFIGURATION = {
    "beam": 20,
    "beam_climbing": True,
    "no_revisit_dets": True,
    "pqlimit": 1000000,
    "order_sources": ["bfs", "coordinate", "index"],
    "orders_per_source": 21,
    "det_order_seed": 518278944,
    "sparsify_errors": False,
}


def decoder_identity() -> dict:
    """Name the decoder, its installed version, and its search settings for reports."""
    load_tesseract()
    version = importlib.metadata.version("tesseract-decoder")
    return {"name": "tesseract", "version": version, **PUBLISHED_CONFIGURATION}


def detector_orders(utils, dem) -> list[list[int]]:
    """Concatenate generated orders per source, as Tesseract's command line does."""
    methods = {
        "bfs": utils.DetOrder.DetBFS,
        "coordinate": utils.DetOrder.DetCoordinate,
        "index": utils.DetOrder.DetIndex,
    }
    order_count = PUBLISHED_CONFIGURATION["orders_per_source"]
    seed = PUBLISHED_CONFIGURATION["det_order_seed"]
    orders = []
    for source in PUBLISHED_CONFIGURATION["order_sources"]:
        method = methods[source]
        source_orders = utils.build_det_orders(dem, order_count, method, seed)
        orders.extend(source_orders)
    return orders


@dataclass
class CleanupPredictions:
    logical: np.ndarray
    unreliable: np.ndarray


@dataclass
class CleanupShot:
    logical: np.ndarray
    unreliable: bool


class TesseractCleanup:
    """Decode Noah's full native hypergraph DEM with his published search settings.

    A low-confidence search counts as a failure, never as a discarded shot.
    """

    def __init__(self, circuit):
        tesseract_package = load_tesseract()
        self.dem = circuit.detector_error_model(approximate_disjoint_errors=True)
        orders = detector_orders(tesseract_package.utils, self.dem)
        configuration = tesseract_package.tesseract.TesseractConfig(
            dem=self.dem,
            det_beam=PUBLISHED_CONFIGURATION["beam"],
            beam_climbing=PUBLISHED_CONFIGURATION["beam_climbing"],
            no_revisit_dets=PUBLISHED_CONFIGURATION["no_revisit_dets"],
            pqlimit=PUBLISHED_CONFIGURATION["pqlimit"],
            det_orders=orders,
            sparsify_errors=PUBLISHED_CONFIGURATION["sparsify_errors"],
        )
        self.decoder = tesseract_package.tesseract.TesseractDecoder(configuration)
        self.version = importlib.metadata.version("tesseract-decoder")

    def decode_one(self, syndrome: np.ndarray) -> CleanupShot:
        """The flag is set when the bounded search ends without explaining the syndrome."""
        prediction = self.decoder.decode(syndrome)
        low_confidence = self.decoder.low_confidence_flag
        return CleanupShot(logical=prediction, unreliable=low_confidence)

    def decode(self, detectors: np.ndarray) -> CleanupPredictions:
        """Decode a batch while retaining every shot's confidence flag."""
        shot_count = len(detectors)
        logical = np.zeros((shot_count, self.dem.num_observables), dtype=bool)
        unreliable = np.zeros(shot_count, dtype=bool)
        for shot_index, syndrome in enumerate(detectors):
            decoded = self.decode_one(syndrome)
            logical[shot_index] = decoded.logical
            unreliable[shot_index] = decoded.unreliable
        return CleanupPredictions(logical=logical, unreliable=unreliable)


def logical_failures(predictions: CleanupPredictions, actual_logical: np.ndarray) -> np.ndarray:
    """Count unreliable cleanup outputs as failures rather than dropping their shots."""
    disagreements = predictions.logical ^ actual_logical
    failures = np.any(disagreements, axis=1)
    failures |= predictions.unreliable
    return failures
