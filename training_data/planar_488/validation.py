"""Held-out model loss and validation-only correction-threshold selection."""

from operator import itemgetter

from training_data.planar_488.evaluation import classification_counts, correction_effect
from training_data.planar_488.evaluation import detector_event_counts
from training_data.planar_488.learning import predict_probabilities, probability_loss
from training_data.planar_488.policy import ABSTENTION_THRESHOLD, abstains


THRESHOLDS = (.5, .7, .8, .9, .95, .99, ABSTENTION_THRESHOLD)
# A learned threshold must reach this empirical label precision on validation.
MINIMUM_PRECISION = .95


def validate_model(model, batches, device, batch_size: int):
    """Masked loss averaged over every valid site of every case, plus its predictions."""
    probabilities_by_case = []
    loss_total = 0.0
    position_total = 0
    for batch in batches:
        probabilities = predict_probabilities(model, batch, device, batch_size)
        probabilities_by_case.append(probabilities)
        loss = probability_loss(batch, probabilities)
        valid_positions = batch.target_mask.sum()
        positions_per_shot = int(valid_positions)
        position_count = positions_per_shot * len(batch.inputs)
        loss_total += loss * position_count
        position_total += position_count
    mean_loss = loss_total / position_total
    return mean_loss, probabilities_by_case


def choose_threshold(batches, probabilities_by_case):
    """Select on validation only, requiring 95% empirical label precision.

    Threshold 1 means abstention. Report that outcome explicitly if no learned
    threshold improves face density. This rule is a pilot heuristic, not a
    statistical guarantee or a logical-error optimization.
    """
    candidates = []
    for threshold in THRESHOLDS:
        true_positive = 0
        false_positive = 0
        original_face_events = 0
        residual_face_events = 0
        for batch, probabilities in zip(batches, probabilities_by_case):
            head_counts = classification_counts(batch, probabilities, threshold)
            for counts in head_counts:
                true_positive += counts["true_positive"]
                false_positive += counts["false_positive"]
            effect = correction_effect(batch, probabilities, threshold)
            event_counts = detector_event_counts(batch, effect)
            original_face_events += event_counts["original_face_events"]
            residual_face_events += event_counts["residual_face_events"]
        positive_predictions = true_positive + false_positive
        precision = None
        if positive_predictions:
            precision = true_positive / positive_predictions
        eligible = abstains(threshold)
        precise_enough = precision is not None and precision >= MINIMUM_PRECISION
        if precise_enough:
            eligible = residual_face_events < original_face_events
        candidates.append({
            "threshold": threshold,
            "precision": precision,
            "positive_predictions": positive_predictions,
            "original_face_events": original_face_events,
            "residual_face_events": residual_face_events,
            "eligible": eligible,
        })
    eligible_candidates = []
    for candidate in candidates:
        if candidate["eligible"]:
            eligible_candidates.append(candidate)
    residual_events = itemgetter("residual_face_events")
    selected = min(eligible_candidates, key=residual_events)
    return selected["threshold"], candidates
