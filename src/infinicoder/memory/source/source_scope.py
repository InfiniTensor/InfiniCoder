"""Hash-guarded Python function work packets within larger source modules."""

import ast, textwrap


def declarations(source):
    result = {}

    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = node
            elif isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")

    visit(ast.parse(source).body)
    return result


def choose(source, task):
    import re

    matches = [
        name
        for name in declarations(source)
        if ("." in name or name.startswith("_"))
        and re.search(r"(?<![\w.])" + re.escape(name) + r"(?!\w|\.[A-Za-z_])", task)
    ]
    return matches[0] if len(matches) == 1 else None


def packet(source, qualname, max_context=6500):
    tree = ast.parse(source)
    owner = tree
    for part in qualname.split(".")[:-1]:
        owner = (
            next((n for n in owner.body if isinstance(n, ast.ClassDef) and n.name == part), None)
            if owner is not None
            else None
        )
    node = (
        next(
            (
                n
                for n in owner.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == qualname.split(".")[-1]
            ),
            None,
        )
        if owner is not None
        else None
    )
    if node is None:
        raise ValueError("unknown Python worker symbol: " + qualname)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    end = node.end_lineno
    lines = source.splitlines(keepends=True)
    original = "".join(lines[start - 1 : end])
    fragment = textwrap.dedent(original)
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    parameters = [p.arg for p in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]]
    parameter_reads = {
        name: sorted(
            {
                n.lineno
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name
            }
        )
        for name in parameters
    }
    statements = [
        n
        for n in node.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]
    is_stub = all(
        isinstance(n, ast.Pass)
        or isinstance(n, ast.Raise)
        and any(isinstance(v, ast.Name) and v.id == "NotImplementedError" for v in ast.walk(n))
        for n in statements
    )
    evidence = []
    size = 0
    class_evidence = []
    owner = tree
    for part in qualname.split(".")[:-1]:
        owner = next(
            (n for n in owner.body if isinstance(n, ast.ClassDef) and n.name == part), None
        )
    if isinstance(owner, ast.ClassDef):
        receiver = node.args.args[0].arg if node.args.args else None
        called = {
            n.func.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == receiver
        }
        members = [
            n
            for n in owner.body
            if n is not node
            and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign))
        ]
        members.sort(
            key=lambda n: (
                0
                if isinstance(n, (ast.Assign, ast.AnnAssign))
                or getattr(n, "name", None) == "__init__"
                else 1
                if getattr(n, "name", None) in called
                else 2,
                n.lineno,
            )
        )
        for member in members:
            if len(class_evidence) >= 32:
                break
            name = getattr(member, "name", "class binding")
            first = min([member.lineno] + [d.lineno for d in getattr(member, "decorator_list", [])])
            exact = (
                not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                or name == "__init__"
                or name in called
            )
            last = member.end_lineno if exact else member.body[0].lineno - 1
            text = "".join(lines[first - 1 : last])
            view = "exact_source" if exact else "declaration_only"
            if not exact and member.body[0].lineno == member.lineno:
                text = (
                    lines[member.lineno - 1]
                    .encode("utf-8")[: member.body[0].col_offset]
                    .decode("utf-8")
                    + " ...\n"
                )
            if exact:
                used.update(
                    n.id
                    for n in ast.walk(member)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                )
            else:
                used.update(n.id for n in ast.walk(member.args) if isinstance(n, ast.Name))
            record = dict(owner=owner.name, name=name, line=first, end=member.end_lineno, view=view)
            if size + len(text) <= max_context:
                record["text"] = text
                size += len(text)
            else:
                record.update(
                    view="location_only", reason="inspect this larger class member on demand"
                )
            class_evidence.append(record)
    for declaration in tree.body:
        names = set()
        if isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(declaration.name)
        elif isinstance(declaration, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in declaration.names)
        elif isinstance(declaration, (ast.Assign, ast.AnnAssign)):
            for target in (
                declaration.targets if isinstance(declaration, ast.Assign) else [declaration.target]
            ):
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        relevant = (
            bool(names & used)
            or is_stub
            and isinstance(
                declaration, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)
            )
        )
        if not relevant or declaration is node or qualname.split(".")[0] in names:
            continue
        first = min(
            [declaration.lineno] + [d.lineno for d in getattr(declaration, "decorator_list", [])]
        )
        text = "".join(lines[first - 1 : declaration.end_lineno])
        view = "exact_source"
        if (
            is_stub
            and isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not names & used
        ):
            text = "".join(lines[first - 1 : declaration.body[0].lineno - 1])
            view = "declaration_only"
        if len(evidence) >= 32:
            continue
        if size + len(text) > max_context:
            evidence.append(
                dict(
                    names=sorted(names),
                    line=first,
                    end=declaration.end_lineno,
                    view="location_only",
                    reason="use source inspection for this larger dependency",
                )
            )
        else:
            evidence.append(
                dict(
                    names=sorted(names),
                    line=first,
                    end=declaration.end_lineno,
                    text=text,
                    view=view,
                )
            )
            size += len(text)
    return dict(
        qualname=qualname,
        start=start,
        end=end,
        indent=node.col_offset,
        source=fragment,
        original=original,
        module_evidence=evidence,
        class_evidence=class_evidence,
        parameter_reads=parameter_reads,
        parameters_without_direct_reads=[name for name in parameters if not parameter_reads[name]],
        limitation="Static syntax only;parameter read lines may include nested scopes. Reflection, dynamic dispatch, inherited state and transitive helper semantics require inspection/independent checks.",
    )


def restore(scope, proposal):
    before = ast.parse(scope["source"]).body
    after = ast.parse(proposal).body
    if len(after) != 1 or not isinstance(after[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError(
            "Scoped worker must return exactly one complete function, with no module-level imports or other declarations"
        )
    old, new = before[0], after[0]

    def signature(node):
        import copy

        args = copy.deepcopy(node.args)
        for arg in (
            [*args.posonlyargs, *args.args, *args.kwonlyargs]
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        ):
            arg.annotation = None
        return (
            type(node).__name__,
            node.name,
            ast.dump(args),
            [ast.dump(p) for p in getattr(node, "type_params", [])],
        )

    if signature(old) != signature(new):
        raise ValueError(
            "Scoped worker changed the protected function signature/decorators;use action mode for interface changes"
        )
    if new.decorator_list and [ast.dump(d) for d in old.decorator_list] != [
        ast.dump(d) for d in new.decorator_list
    ]:
        raise ValueError("Scoped worker changed protected decorators")
    # Keep the exact original declaration (including annotations/decorators).
    # Merge only the proposed body, normalizing its indentation structurally.
    old_lines = scope["source"].splitlines(keepends=True)
    prefix = "".join(old_lines[: old.body[0].lineno - 1])
    body_indent = old.body[0].col_offset
    if old.body[0].lineno == old.lineno:
        prefix += (
            old_lines[old.lineno - 1]
            .encode("utf-8")[: old.body[0].col_offset]
            .decode("utf-8")
            .rstrip()
            + "\n"
        )
        body_indent = old.col_offset + 4
    new_lines = proposal.splitlines(keepends=True)[new.body[0].lineno - 1 : new.end_lineno]
    if new.body[0].lineno == new.lineno:
        new_lines[0] = new_lines[0].encode("utf-8")[new.body[0].col_offset :].decode("utf-8")
    body = "".join(new_lines)
    result = prefix + textwrap.indent(textwrap.dedent(body), " " * body_indent)
    return textwrap.indent(result, " " * scope["indent"])
