"""AFD (Attention-FFN Disaggregation) role helpers.

The AFD role is derived from `disaggregation_mode`:
    - "prefill" / "decode" -> "attn"   (this server holds attention; experts live elsewhere)
    - "expert"             -> "ffn"    (this server holds experts; attention lives elsewhere)
    - "null"               -> None     (monolithic; no AFD)

PD disaggregation and AFD coexist by sharing this single mode flag. There is no
separate --afd-role; the user picks one of {prefill, decode, expert} via
--disaggregation-mode.
"""

from typing import Optional


def get_afd_role() -> Optional[str]:
    from sglang.srt.server_args import get_global_server_args

    try:
        sa = get_global_server_args()
    except ValueError:
        # Server args not initialized yet (e.g., during tokenizer side import).
        return None
    mode = getattr(sa, "disaggregation_mode", None)
    if mode in ("prefill", "decode"):
        return "attn"
    if mode == "expert":
        return "ffn"
    return None


def afd_is_attn() -> bool:
    return get_afd_role() == "attn"


def afd_is_ffn() -> bool:
    return get_afd_role() == "ffn"


def afd_is_active() -> bool:
    return get_afd_role() is not None
