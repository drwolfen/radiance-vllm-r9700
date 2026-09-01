#!/usr/bin/env python3
"""Bake: fix streaming vs non-streaming tool-parser divergence on a truncated
tool call (vLLM issue #47137), for the engine-based parsers.

Content leak (abstract_parser.py) is merged in vLLM 0.28 (PR #46875) -> optional=True.
Args divergence (parser_engine.py) remains unmerged in 0.28.0 -> applied.
"""
import ast
import sysconfig
from pathlib import Path
from _patchlib import apply

LIB = Path(sysconfig.get_paths()["purelib"])

# ── 1. content-leak fix (abstract_parser.py) ─────────────────────────────────
F_ABS = LIB / "vllm/parser/abstract_parser.py"
ABS_ANCHOR = (
    "                if (is_required_tool_choice or is_named_tool_choice) and (\n"
    "                    content is None\n"
    "                    or (isinstance(content, str) and not content.strip())\n"
    "                ):\n"
    "                    return [], None\n"
    "                return None, content\n"
)
ABS_NEW = (
    "                if (is_required_tool_choice or is_named_tool_choice) and (\n"
    "                    content is None\n"
    "                    or (isinstance(content, str) and not content.strip())\n"
    "                ):\n"
    "                    return [], None\n"
    "                # Engine-based parsers already strip incomplete / un-promoted\n"
    "                # tool-call markup from content, so return that (drops it)\n"
    "                # rather than the raw input. Keeps non-streaming in agreement\n"
    "                # with streaming on a truncated <tool_call> opener (vLLM #47137).\n"
    "                if self._engine_based and tool_call_info is not None:\n"
    "                    return None, tool_call_info.content\n"
    "                return None, content\n"
)
ABS_SENTINEL = "if self._engine_based and tool_call_info is not None:\n                    return None, tool_call_info.content"

# ── 2. args-divergence fix (parser_engine.py) ────────────────────────────────
F_ENG = LIB / "vllm/parser/engine/parser_engine.py"
ENG_ANCHOR = (
    "                    try:\n"
    "                        args_json = converter(raw_body, False)\n"
    "                    except (json.JSONDecodeError, ValueError, TypeError):\n"
    "                        logger.debug(\n"
    '                            "arg converter failed (extract): %s", raw_body[:80]\n'
    "                        )\n"
    "                        args_json = self._extract_args_json(raw_body, name)\n"
)
ENG_NEW = (
    "                    try:\n"
    "                        args_json = converter(raw_body, False)\n"
    "                    except (json.JSONDecodeError, ValueError, TypeError):\n"
    "                        logger.debug(\n"
    '                            "arg converter failed (extract): %s", raw_body[:80]\n'
    "                        )\n"
    "                        args_json = self._extract_args_json(raw_body, name)\n"
    "                    else:\n"
    "                        # vLLM #47137: a call truncated mid-parameter drops\n"
    "                        # the unterminated value under partial=False, but a\n"
    "                        # streaming client already received it. When\n"
    "                        # partial=True parses further, the body is truncated;\n"
    "                        # emit the exact streamed args so non-streaming agrees\n"
    "                        # with streaming.\n"
    "                        try:\n"
    "                            partial_json = converter(raw_body, True)\n"
    "                        except (json.JSONDecodeError, ValueError, TypeError):\n"
    "                            partial_json = args_json\n"
    "                        if partial_json != args_json and slot.streamed_json:\n"
    "                            args_json = slot.streamed_json\n"
)
ENG_SENTINEL = "if partial_json != args_json and slot.streamed_json:\n                            args_json = slot.streamed_json"


def main():
    apply(F_ABS, ABS_ANCHOR, ABS_NEW, ABS_SENTINEL, "tool-content-leak", optional=True)
    apply(F_ENG, ENG_ANCHOR, ENG_NEW, ENG_SENTINEL, "tool-args-truncation", optional=False)


if __name__ == "__main__":
    main()
