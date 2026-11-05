from __future__ import annotations

import os


SFP_GENTLE_REFERENCE_TUNINGS = frozenset(
    {
        "gentle_ref_pack_v1",
        "reference_gentle_v1",
        "sfp_gentle_ref_pack_v1",
    }
)


def get_sfp_primitive_tuning_profile() -> str:
    value = os.environ.get(
        "AIC_ACT_SFP_PRIMITIVE_TUNING",
        os.environ.get("AIC_ACT_SFP_PRIMITIVE_VARIANT", "default"),
    )
    return (value or "default").strip().lower() or "default"


def is_sfp_gentle_reference_tuning(profile: str | None = None) -> bool:
    return (
        get_sfp_primitive_tuning_profile() if profile is None else profile
    ) in SFP_GENTLE_REFERENCE_TUNINGS
