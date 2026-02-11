TRANSLATIONS = {
    "en": {
        "app_title": "Intelligent Weapon Detection System",
        "upload_header": "Upload Image or Video",
        "detect_button": "Detect Weapons",
        "results_header": "Detection Results",
        "no_weapons": "No weapons detected.",
        "weapons_detected": "Weapons Detected!",
        "total_count": "Total Weapons",
        "summary": "Summary",
        "confidence": "Confidence",
    },
    "te": {
        "app_title": "తెలివైన ఆయుధ గుర్తింపు వ్యవస్థ",
        "upload_header": "చిత్రం లేదా వీడియోను అప్‌లోడ్ చేయండి",
        "detect_button": "ఆయుధాలను గుర్తించండి",
        "results_header": "గుర్తింపు ఫలితాలు",
        "no_weapons": "ఆయుధాలు కనుగొనబడలేదు.",
        "weapons_detected": "ఆయుధాలు గుర్తించబడ్డాయి!",
        "total_count": "మొత్తం ఆయుధాలు",
        "summary": "సారాంశం",
        "confidence": "ఖచ్చితత్వం",
    }
}

def get_text(key, lang="en"):
    """
    Retrieves the translated text for a given key and language.
    """
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
