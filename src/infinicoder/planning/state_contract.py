"""Conservative source-bound checks of a model's own explicit state claims."""

import re
from infinicoder.runtime.worker import _native_mask
from infinicoder.memory.state_memory import close

DESIGN_GUIDANCE = """
State transition design:use visible initializer clauses to establish conditional
initial state;loads and later writes can override it. Include state_effects as
an array (empty if no applicable direct scalar update). For an explicit scalar
increment/decrement,bind field to a visible field ID/name;operation is increment
or decrement;placement is outside_loops for one logical-step update or inside_loop
for per-item update;order is before_calls or after_calls relative to direct calls
receiving that member as an argument. These are model claims,not semantic proof.
Separate parameter/element state from whole-invocation state. Check initial
values and consumer formulas before choosing update order."""

CODE_GUIDANCE = """
Preserve the design's explicit state_effects or explain a source-based revision.
Before returning source,compare each update's loop placement and order with
the actual call arguments. A proven syntax-order conflict is returned before
editing or integration testing. Aliases/complex control remain unknown.
After conflict feedback,you may return one complete revised design JSON with
read_ids,state_targets,invariant,steps,state_effects if the hypothesis was wrong.
Any need_context object must contain your precise missing symbol or source
question;never copy a generic placeholder. Original independent acceptance
still decides correctness."""


def effect_schema():
    return dict(
        type="array",
        maxItems=6,
        items=dict(
            type="object",
            properties=dict(
                field=dict(type="string", minLength=1, maxLength=200),
                operation=dict(type="string", enum=["increment", "decrement"]),
                placement=dict(type="string", enum=["outside_loops", "inside_loop"]),
                order=dict(type="string", enum=["before_calls", "after_calls"]),
            ),
            required=["field", "operation", "placement", "order"],
            additionalProperties=False,
        ),
    )


def bind(effects, reference):
    if not isinstance(effects, list) or len(effects) > 6:
        raise ValueError("Bounded state_effects list required")
    result = []
    for effect in effects:
        if not isinstance(effect, dict) or set(effect) != {
            "field",
            "operation",
            "placement",
            "order",
        }:
            raise ValueError("Typed state effect claim required")
        candidates = [
            r
            for r in reference["fields"]
            if effect["field"] in (r["id"], r["name"], r["short_name"])
        ]
        if len(candidates) != 1:
            raise ValueError("State claim field is not a unique current visible source field")
        if (
            effect["operation"] not in ("increment", "decrement")
            or effect["placement"] not in ("outside_loops", "inside_loop")
            or effect["order"] not in ("before_calls", "after_calls")
        ):
            raise ValueError("Unsupported state claim")
        if any(r["field_id"] == candidates[0]["id"] for r in result):
            raise ValueError("One state effect claim per field required")
        result.append(
            dict(
                effect,
                field_id=candidates[0]["id"],
                field_name=candidates[0]["short_name"],
                claim_not_proof=True,
            )
        )
    return result


def check(source, effects):
    from infinicoder.execution.native_scope import _header

    if len(source) > 32768:
        return dict(
            status="unknown",
            claims=[],
            violations=[],
            semantic_proof=False,
            model_calls=0,
            limitations=["Source exceeds bounded state preflight size."],
        )
    opening, end, _ = _header(source)
    mask = _native_mask(source)
    body = mask[opening + 1 : end]
    base = opening + 1
    report = dict(
        status="unknown",
        claims=[],
        violations=[],
        semantic_proof=False,
        model_calls=0,
        limitations=[
            "Syntax placement/order only;not mutation-by-callee,alias,exception or dynamic-dispatch proof.",
            "Consistent model claims may be semantically wrong;original independent acceptance remains required.",
        ],
    )
    # Do not pretend to reason about jumps,lambda/local types or complex control.
    unsupported = bool(
        re.search(
            r"\b(goto|switch|do|try|catch|co_await|co_yield|class|struct)\b|\[[^\]]*\]\s*(?:\([^)]*\))?\s*(?:mutable\s*)?\{",
            body,
        )
    )
    loops = []
    conditions = []
    for match in re.finditer(r"\b(for|while|if)\s*\(", body):
        at = body.find("(", match.start())
        try:
            right = close(body, at) + 1
        except ValueError:
            unsupported = True
            continue
        while right < len(body) and body[right].isspace():
            right += 1
        if right >= len(body) or body[right] != "{":
            unsupported = True
            continue
        finish = close(body, right)
        span = (base + match.start(), base + finish + 1)
        (conditions if match.group(1) == "if" else loops).append(span)
    if re.search(r"\belse\b", body):
        unsupported = True
    calls = []
    grammar = r"\b[A-Za-z_]\w*(?:(?:::|\.|->)[A-Za-z_]\w*)*(?:\s*<[^;{}()]*>)?\s*\("
    for match in re.finditer(grammar, body):
        spelling = match.group().rstrip()[:-1].strip()
        leaf = re.split(r"::|\.|->", spelling)[-1].split("<")[0]
        if leaf in (
            "if",
            "for",
            "while",
            "sizeof",
            "decltype",
            "alignof",
            "static_cast",
            "reinterpret_cast",
            "const_cast",
            "dynamic_cast",
        ):
            continue
        at = body.find("(", match.start())
        try:
            finish = close(body, at)
        except ValueError:
            continue
        arguments = []
        start = at + 1
        position = start
        while position < finish:
            c = body[position]
            if c in "({[":
                position = close(body, position) + 1
                continue
            if c == ",":
                arguments.append(body[start:position].strip())
                start = position + 1
            position += 1
        arguments.append(body[start:finish].strip())
        calls.append(dict(expression=spelling, position=base + match.start(), arguments=arguments))
    for claim in effects:
        name = claim["field_name"]
        escaped = re.escape(name)
        member = r"(?:this\s*->\s*)?\b" + escaped + r"\b"
        detail = dict(field_id=claim["field_id"], field_name=name, status="unknown", claim=claim)
        # A same-named parameter/local binding would need semantic resolution.
        shadow = name in re.findall(r"\b[A-Za-z_]\w*\b", mask[:opening]) or bool(
            re.search(
                r"\b(?:auto|const|volatile|[A-Za-z_]\w*(?:::\w+)*(?:<[^;{}]+>)?)\s+[*&\s]*"
                + escaped
                + r"\b\s*(?:=|;|\{|,)",
                body,
            )
        )
        updates = []
        for match in re.finditer(
            r"(?<![\w.>])(?:(\+\+|--)\s*" + member + r"|" + member + r"\s*(\+\+|--))", body
        ):
            updates.append(
                dict(position=base + match.start(), operator=match.group(1) or match.group(2))
            )
        consumers = [c for c in calls if any(re.fullmatch(member, a) for a in c["arguments"])]
        detail.update(
            updates=updates,
            consumers=[{k: v for k, v in c.items() if k != "arguments"} for c in consumers],
        )
        assignments = bool(re.search(member + r"\s*(?:=(?!=)|[+*/%&|^\-]=)", body))
        if unsupported or shadow or assignments or len(updates) != 1 or not consumers:
            detail["reason"] = (
                "Unsupported control,shadow/assignment,nonunique update or no direct member-argument consumer"
            )
        else:
            update = updates[0]
            nest = [s for s in loops if s[0] < update["position"] < s[1]]
            conditional = any(a < update["position"] < b for a, b in conditions)
            if conditional:
                detail["reason"] = "Conditional update needs path-sensitive analysis"
            elif re.search(r"\b(return|throw)\b", mask[base : update["position"]]):
                detail["reason"] = "Early exit may bypass update"
            else:
                bad = []
                if update["operator"] != ("++" if claim["operation"] == "increment" else "--"):
                    bad.append("update_operation")
                if bool(nest) != (claim["placement"] == "inside_loop"):
                    bad.append("loop_placement")
                comparable = [
                    c for c in consumers if not nest or all(a < c["position"] < b for a, b in nest)
                ]
                if not comparable:
                    detail["reason"] = "Update and uses have different loop scopes"
                else:
                    before = all(update["position"] < c["position"] for c in comparable)
                    after = all(update["position"] > c["position"] for c in comparable)
                    if not (before if claim["order"] == "before_calls" else after):
                        bad.append("update_use_order")
                    detail.update(
                        status="conflict" if bad else "structurally_consistent", conflicts=bad
                    )
                    for kind in bad:
                        report["violations"].append(
                            dict(
                                field_id=claim["field_id"],
                                field_name=name,
                                kind=kind,
                                update_position=update["position"],
                                use_positions=[c["position"] for c in comparable],
                            )
                        )
        report["claims"].append(detail)
    report["status"] = (
        "conflict"
        if report["violations"]
        else "structurally_consistent"
        if effects and all(r["status"] == "structurally_consistent" for r in report["claims"])
        else "unknown"
    )
    return report
