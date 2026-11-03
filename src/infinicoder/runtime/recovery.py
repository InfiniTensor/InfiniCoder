"""Evidence-based recovery across changing proposals, without task dispatch."""

import hashlib
import json
import re
from infinicoder.runtime.feedback import failure_packet


class RecoveryPolicy:
    def observe(self, result, state):
        rows = failure_packet(result)
        if not rows:
            return None
        row = rows[0]
        diagnostics = row.get("diagnostics") or []
        text = str(
            row.get("error")
            or row.get("detail")
            or row.get("rejected_because")
            or row.get("reason")
            or ""
        )
        command_protocol = "commandargvformaterror" in text.lower()
        if result.get("failure_origin") == "response_protocol":
            category, guidance = (
                "protocol",
                "The runtime rejected the response before action dispatch. Follow the current SYSTEM transport and advertised tool/argument schema;no target implementation was executed by this action. Inspect current source as needed.",
            )
            symptom = text
        elif result.get("failure_origin") == "design_contract":
            category, guidance = (
                "protocol",
                "Repair the design parameters or current source lease before changing implementation. A malformed symbolic probe is not an independent code failure.",
            )
            symptom = text
        elif result.get("failure_origin") == "tool_navigation":
            category, guidance = (
                "navigation",
                "Use observed current paths or unread source ranges. A missing source path is a navigation failure;do not change the implementation or create an invented module to repair the tool call.",
            )
            symptom = "source_path_not_found"
        elif command_protocol:
            category, guidance = (
                "protocol",
                'Fix the command argument vector before changing source. Pass the executable and each argument as separate strings;for shell syntax explicitly use ["bash","-c","SCRIPT"] on Linux,keeping the current environment. The working directory is already the workspace.',
            )
            symptom = "command_argument_vector_format"
        elif diagnostics or "compile failed" in text.lower() or "syntaxerror" in text.lower():
            category, guidance = (
                "compilation",
                "Repair the first compiler error; preserve the interface and avoid unrelated changes.",
            )
            symptom = diagnostics[0].get("message", text) if diagnostics else text.splitlines()[0]
        elif (
            "sourceformaterror" in text.lower()
            or "json" in text.lower()
            or "incomplete model output" in text.lower()
        ):
            category, guidance = (
                "protocol",
                "Use the currently advertised response format with complete physical source lines and indentation. Keep the required interface and omit reasoning or extra transport wrappers.",
            )
            symptom = text
        elif row.get("exit_code") == 124:
            category, guidance = (
                "resource",
                "Check loop progress and blocking synchronization before changing timeout.",
            )
            symptom = "process_timeout"
        else:
            category, guidance = (
                "semantic",
                "Compare the failing input with the declared contract; distinguish valid boundary inputs from invalid inputs.",
            )
            symptom = text
        # Values and snapshot ids vary even when the same root observation repeats.
        normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "#", symptom[:1500])
        key = hashlib.sha256(
            json.dumps([row.get("check_id"), category, normalized]).encode()
        ).hexdigest()
        counts = state.setdefault("recovery_failures", {})
        count = counts.get(key, 0) + 1
        counts[key] = count
        if count >= 2 and category == "semantic":
            domain = state.get("route", {}).get("profile")
            if domain == "operator":
                guidance += " If framework tensor primitives are permitted, use documented operations and explicit dimensions; inspect API signatures instead of inventing keywords. Otherwise simplify device indexing and data dependencies. Recheck correctness before optimization."
            elif domain == "framework":
                guidance += " Inspect actual argument types, interfaces and state transitions; preserve synchronization and checkpoint behavior before optimizing."
            else:
                guidance += " Simplify to explicit indexing, scalar loops and explicit state transitions; establish correctness before optimization."
        if count >= 3:
            guidance += (
                " Stop resending the same shell string;correct its argument structure without changing the task contract."
                if command_protocol
                else " Correct the active response packaging before another source proposal."
                if category == "protocol"
                else " Previous proposals share this symptom: derive the implementation again from the contract instead of making another cosmetic edit."
            )
        return {"category": category, "count": count, "signature": key, "guidance": guidance}
