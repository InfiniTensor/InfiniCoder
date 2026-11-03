"""One audited JSON delimiter repair;never rewrites model strings or source."""

import hashlib
import json


def load(text, *, finish_reason=None):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as original:
        if finish_reason == "length" or not text.endswith("}") or original.pos != len(text) - 1:
            raise
        stack = []
        quoted = False
        escaped = False
        for char in text[:-1]:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or stack.pop() != ("{" if char == "}" else "["):
                    raise original
        if quoted or stack != ["{", "["]:
            raise original
        repaired = text[:-1] + "]}"
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            raise original
        if not isinstance(value, dict):
            raise original
        return value, dict(
            operation="insert_one_final_array_closure",
            position=len(text) - 1,
            original_sha256=hashlib.sha256(text.encode()).hexdigest(),
            normalized_sha256=hashlib.sha256(repaired.encode()).hexdigest(),
            scalar_text_changed=False,
            semantic_validation=False,
        )
