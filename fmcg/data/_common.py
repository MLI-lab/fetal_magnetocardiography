"""Shared constants for the WFDB loaders."""

# Annotation symbols that mark actual heartbeat positions (R-peaks). Shared by every
# PhysioNet loader so the beat-vs-nonbeat filter is defined in exactly one place.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
