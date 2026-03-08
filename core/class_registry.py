"""
Canonical Class Registry — Single source of truth for class definitions.

All class IDs across models, datasets, ensemble fusion, and UI rendering
must resolve through this registry. No other module should define class
names or IDs independently.
"""

# Canonical class ID → name mapping.
# This is the authoritative mapping used after normalization.
CANONICAL_CLASSES = {
    0: "guns",
    1: "knife",
}

# Canonical name → class ID (reverse of CANONICAL_CLASSES).
CANONICAL_CLASS_TO_ID = {name: cid for cid, name in CANONICAL_CLASSES.items()}

# Synonym map: maps any model-specific label string to its canonical name.
# Add entries here when integrating models trained with non-standard labels.
SYNONYM_MAP = {
    "pistol": "guns",
    "gun": "guns",
    "guns": "guns",
    "weapon": "guns",   # DatasetNinja label
    "knife": "knife",
}

# Labels considered weapons for UI color coding and threat status.
WEAPON_LABELS = frozenset(CANONICAL_CLASS_TO_ID.keys())
