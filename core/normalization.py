"""
Prediction Normalization Layer.

Converts model-specific class IDs to canonical IDs using the model's
names dict and the central SYNONYM_MAP. This MUST be applied between
model inference and WBF fusion — never after.
"""

from core.class_registry import SYNONYM_MAP, CANONICAL_CLASS_TO_ID


def normalize_class_id(model_class_id: int, model_names: dict) -> tuple[int, str]:
    """
    Maps a model-specific class ID to its canonical class ID and name.

    Args:
        model_class_id: The integer class ID from the model's prediction.
        model_names: The model's {id: name} mapping (from model.names).

    Returns:
        (canonical_id, canonical_name) tuple.

    Raises:
        KeyError: If the model's label string is not in SYNONYM_MAP.
    """
    model_label = model_names.get(model_class_id, str(model_class_id))
    canonical_name = SYNONYM_MAP.get(model_label.lower())

    if canonical_name is None:
        # Unknown class — pass through with a warning.
        # This avoids silent failures but surfaces unexpected labels.
        import warnings
        warnings.warn(
            f"Unknown class label '{model_label}' (id={model_class_id}) "
            f"not found in SYNONYM_MAP. Skipping detection.",
            stacklevel=2,
        )
        return -1, model_label

    canonical_id = CANONICAL_CLASS_TO_ID[canonical_name]
    return canonical_id, canonical_name


def normalize_detections(boxes, scores, labels, model_names: dict):
    """
    Normalizes a batch of detections from a single model to canonical IDs.

    Filters out any detections whose labels are not in the SYNONYM_MAP
    (e.g., non-weapon classes like 'billete', 'smartphone', 'monedero').

    Args:
        boxes: List of [x1, y1, x2, y2] (normalized 0-1).
        scores: List of float confidence scores.
        labels: List of int model-specific class IDs.
        model_names: The model's {id: name} mapping.

    Returns:
        (norm_boxes, norm_scores, norm_labels) with canonical IDs only.
    """
    norm_boxes, norm_scores, norm_labels = [], [], []

    for box, score, label_id in zip(boxes, scores, labels):
        canonical_id, _ = normalize_class_id(label_id, model_names)
        if canonical_id == -1:
            continue  # Skip unknown/non-weapon classes
        norm_boxes.append(box)
        norm_scores.append(score)
        norm_labels.append(canonical_id)

    return norm_boxes, norm_scores, norm_labels
