"""Public sequential repair stage names, with legacy history compatibility."""

REPAIR_STAGE_NAMES = {"A": "R1", "B": "R2", "R2": "R3"}


def repair_stage(legacy_strategy: str | None) -> str | None:
    """Translate legacy strategy identifiers; public R2 must not be passed here."""
    value = str(legacy_strategy or "")
    if value.startswith("A_"):
        return "R1"
    if value.startswith("B_"):
        return "R2"
    if value.startswith("round_2"):
        return "R3"
    return REPAIR_STAGE_NAMES.get(value)
