"""Bounded optional design grammar;syntax never certifies a semantic design."""


def schema(state_contract=False, delta_id=None):
    string = lambda maximum: dict(type="string", minLength=1, maxLength=maximum)
    design = dict(
        read_ids=dict(type="array", minItems=1, maxItems=32, items=string(500)),
        state_targets=dict(type="array", maxItems=24, items=string(500)),
        invariant=string(1800),
        steps=dict(type="array", minItems=1, maxItems=8, items=string(600)),
    )
    if state_contract:
        from infinicoder.planning.state_contract import effect_schema

        design["state_effects"] = effect_schema()
    question = dict(need_context=string(1500))
    alternatives = [design, question]
    if delta_id is not None:
        alternatives.append(
            dict(
                reuse_design=dict(type="string", const=delta_id),
                read_ids=design["read_ids"],
                state_targets=design["state_targets"],
            )
        )
    return dict(
        anyOf=[
            dict(type="object", properties=value, required=list(value), additionalProperties=False)
            for value in alternatives
        ]
    )
