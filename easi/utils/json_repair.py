"""JSON repair for LLM output. Ported from EmbodiedBench planner_utils.py."""
from __future__ import annotations

import re


def fix_json(json_str: str) -> str:
    """Fix common JSON errors in LLM output.

    Handles:
    - Single quotes -> double quotes
    - Broken contractions from quote replacement
    - Markdown code fences
    - Unescaped quotes inside reasoning_and_reflection value
    """
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    # Fix contractions that got broken by quote replacement
    json_str = json_str.replace('"s ', "'s ")
    json_str = json_str.replace('"re ', "'re ")
    json_str = json_str.replace('"ll ', "'ll ")
    json_str = json_str.replace('"t ', "'t ")
    json_str = json_str.replace('"d ', "'d ")
    json_str = json_str.replace('"m ', "'m ")
    json_str = json_str.replace('"ve ', "'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # Fix unescaped double quotes inside reasoning_and_reflection value.
    # Pattern: match from the key's opening quote to just before "language_plan".
    pattern = r'("reasoning_and_reflection"\s*:\s*")(?P<value>.*?)(?=",\s*"language_plan")'

    def replacer(match):
        prefix = match.group(1)
        value = match.group("value")
        # Escape any double quote that is not already escaped.
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value

    json_str = re.sub(pattern, replacer, json_str, flags=re.DOTALL)
    return json_str
