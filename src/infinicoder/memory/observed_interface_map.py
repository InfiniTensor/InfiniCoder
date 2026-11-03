"""Represent already delivered source relations;never infer edit permission.

The caller establishes actual message/source association. This builder only
uses complete observed Python definitions and the existing index's unique
static call candidates. It neither discovers files nor evaluates source code.
"""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

from infinicoder.memory.task_memory import TaskMemory

IDENTITY = "observed_interface_map_v1"
RENDERER = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObservedSource:
    path: str
    text: str
    delivered_message_ref: str
    observed_lines: tuple = ()

    def __post_init__(self):
        p = PurePosixPath(self.path)
        if (
            not self.path
            or str(p) != self.path
            or p.is_absolute()
            or ".." in p.parts
            or any(c in self.path for c in ("\\", ":", "\0"))
        ):
            raise ValueError("normalized relative observed path required")
        if not isinstance(self.text, str) or len(self.text.encode("utf-8")) > 1_000_000:
            raise ValueError("bounded exact source required")
        if (
            not isinstance(self.delivered_message_ref, str)
            or not 0 < len(self.delivered_message_ref) <= 200
        ):
            raise ValueError("actual delivered message reference required")
        for first, last in self.observed_lines:
            if (
                type(first) is not int
                or type(last) is not int
                or not 1 <= first <= last <= len(self.text.splitlines())
            ):
                raise ValueError("observed source range is invalid")

    @property
    def source_sha256(self):
        return digest(self.text)

    def covers(self, first, last):
        return 1 <= first <= last <= len(self.text.splitlines()) and (
            not self.observed_lines or any(a <= first <= last <= b for a, b in self.observed_lines)
        )

    def support(self, first, last, role):
        if not self.covers(first, last):
            raise ValueError("support was not completely delivered")
        return dict(
            role=role,
            path=self.path,
            sha=self.source_sha256,
            lines=[first, last],
            text="\n".join(self.text.splitlines()[first - 1 : last]),
            message_ref=self.delivered_message_ref,
        )


@dataclass(frozen=True)
class InterfaceMap:
    payload: dict
    text: str
    tokens: int
    source_versions: dict

    def current(self, current_version):
        try:
            return all(
                current_version(path) == version for path, version in self.source_versions.items()
            )
        except (OSError, ValueError, KeyError):
            return False


def definitions(text):
    found = {}

    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                found[name] = node
                visit(node.body, name + ".")
            elif isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")

    visit(ast.parse(text).body)
    return found


def blocks(nodes):
    """Same lexical statement blocks; do not cross into nested definitions."""
    yield nodes
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for name in ("body", "orelse", "finalbody"):
            value = getattr(node, name, None)
            if isinstance(value, list) and value:
                yield from blocks(value)
        for handler in getattr(node, "handlers", ()):
            yield from blocks(handler.body)


def local_nodes(statement):
    pending = [statement]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


def stored(statement, name):
    return any(
        isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del))
        for n in local_nodes(statement)
    )


def use_of(statement, name):
    # A conditional consumer is recorded as syntax, never as guaranteed runtime
    # execution or as a binding to a same-named receiver method.
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    for node in local_nodes(statement):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
            and isinstance(node.ctx, ast.Load)
        ):
            return dict(kind="indexed_result", expression=ast.unparse(node), line=node.lineno)
        if isinstance(node, ast.Call) and any(
            isinstance(a, ast.Name) and a.id == name
            for a in [*node.args, *(k.value for k in node.keywords)]
        ):
            return dict(
                kind="result_passed_to_call",
                expression=ast.unparse(node.func),
                line=node.lineno,
                consumer_dispatch="unknown",
            )
    return None


def import_support(source, expression):
    """The static cross-file binding itself must also have been observed."""
    root = expression.split(".", 1)[0]
    for node in ast.parse(source.text).body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            name = alias.asname or (
                alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name
            )
            if name == root and source.covers(node.lineno, node.end_lineno):
                return source.support(node.lineno, node.end_lineno, "static_import_binding")
    return None


def build_interface_map(
    index,
    observed,
    *,
    public_contract_ref,
    namespace,
    current_version,
    count_text,
    remaining,
    anchors=(),
    scope_quotes=(),
):
    """Return at most2 relations/6 statements,or None without changing inputs.

    ``count_text`` must use the real tokenizer;no character heuristic is
    substituted here. Runtime integration must also count the whole projected
    messages and enforce the separate five/ten-second maintenance limits.
    """
    if (
        not isinstance(namespace, dict)
        or not namespace.get("environment")
        or not namespace.get("root")
    ):
        raise ValueError("actual source namespace required")
    if not isinstance(public_contract_ref, str) or not public_contract_ref:
        raise ValueError("public contract reference required")
    sources = {item.path: item for item in observed}
    if len(sources) != len(observed) or len(sources) > 32:
        raise ValueError("distinct bounded observed sources required")
    if len(sources) < 2:
        return None
    nodes = {}
    current = {}
    for path, source in sources.items():
        remaining()
        if not path.endswith(".py"):
            continue
        try:
            current_hash = current_version(path)
        except (OSError, ValueError, KeyError):
            continue
        if current_hash != source.source_sha256:
            continue
        row = index.db.execute("SELECT hash,error FROM files WHERE path=?", (path,)).fetchone()
        if row is None or row["hash"] != source.source_sha256 or row["error"]:
            continue
        try:
            nodes[path] = definitions(source.text)
        except SyntaxError:
            continue
        current[path] = source
    memory = TaskMemory(index)
    relations = []
    seen = set()
    wanted = set(anchors)
    symbols = [
        dict(row)
        for row in index.db.execute(
            "SELECT * FROM symbols WHERE kind='function' ORDER BY path,line"
        )
        if row["path"] in current
    ]
    symbols.sort(key=lambda row: (row["id"] not in wanted, row["path"], row["line"]))
    for symbol in symbols[:64]:
        remaining()
        source = current[symbol["path"]]
        function = nodes[source.path].get(symbol["qualname"])
        if function is None or not source.covers(function.lineno, function.end_lineno):
            continue
        summary = memory.summary(symbol["id"])
        links = {
            (link["call_line"], link["call_column"]): link
            for link in summary.get("input_output_links", [])
            if link.get("status") == "static_candidate"
        }
        for body in blocks(function.body):
            for offset, statement in enumerate(body):
                remaining()
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)) or not isinstance(
                    statement.value, ast.Call
                ):
                    continue
                call = statement.value
                link = links.get((call.lineno, call.col_offset))
                if (
                    not link
                    or link["callee_path"] == source.path
                    or link["callee_path"] not in current
                ):
                    continue
                imported = import_support(source, link["target_expression"])
                if imported is None:
                    continue
                callee_source = current[link["callee_path"]]
                callee_symbol = index.symbol(link["callee"])
                callee = nodes[callee_source.path].get(callee_symbol["qualname"])
                if callee is None or not callee_source.covers(callee.lineno, callee.end_lineno):
                    continue
                if callee_source.source_sha256 != link["callee_source_sha256"]:
                    continue
                target = (
                    statement.targets[0]
                    if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    else statement.target
                    if isinstance(statement, ast.AnnAssign)
                    else None
                )
                use = None
                consumer_statement = None
                if isinstance(target, (ast.Tuple, ast.List)) and not any(
                    isinstance(n, ast.Starred) for n in ast.walk(target)
                ):
                    use = dict(
                        kind="unpacked_result", arity=len(target.elts), line=statement.lineno
                    )
                elif isinstance(target, ast.Name):
                    for following in body[offset + 1 :]:
                        # Even a conditional rebind makes this elementary link
                        # unavailable;we do not pretend to have a full CFG.
                        if stored(following, target.id):
                            break
                        use = use_of(following, target.id)
                        if use is not None:
                            consumer_statement = following
                            break
                if use is None:
                    continue
                key = (source.path, call.lineno, link["callee"], dump(use))
                if key in seen:
                    continue
                seen.add(key)
                header_end = max(callee.lineno, callee.body[0].lineno - 1)
                support = [
                    imported,
                    source.support(statement.lineno, statement.end_lineno, "call_result_binding"),
                    callee_source.support(callee.lineno, header_end, "callee_interface"),
                ]
                if consumer_statement is not None:
                    support.append(
                        source.support(
                            consumer_statement.lineno,
                            consumer_statement.end_lineno,
                            "subsequent_use",
                        )
                    )
                relations.append(
                    dict(
                        public_anchor=symbol["id"],
                        callee=link["callee"],
                        use=use,
                        support=support,
                        conditional_binding="Unique static import/function candidate;only if these operations execute without rebinding,dispatch replacement or exceptions.",
                        unknowns=[
                            "runtime values",
                            "intervening heap mutation",
                            "dynamic dispatch",
                            "alias/ownership",
                            "all-path or concurrency guarantees",
                        ],
                    )
                )
                if len(relations) == 2:
                    break
            if len(relations) == 2:
                break
        if len(relations) == 2:
            break
    if not relations:
        return None
    if sum(len(r["support"]) for r in relations) > 6:
        relations = relations[:1]
    for limit, chosen in ((768, relations[:2]), (384, relations[:1])):
        remaining()
        # Share file/version/message identity rather than repeating long hashes
        # at every statement. Every complete support statement is retained.
        files = {}
        normalized = []
        for relation in chosen:
            row = dict(relation)
            row["support"] = []
            for support in relation["support"]:
                ref = next((k for k, v in files.items() if v["path"] == support["path"]), None)
                if ref is None:
                    ref = "f" + str(len(files))
                    files[ref] = {k: support[k] for k in ("path", "sha", "message_ref")}
                row["support"].append(
                    dict(file=ref, **{k: support[k] for k in ("role", "lines", "text")})
                )
            normalized.append(row)
        payload = dict(
            kind=IDENTITY,
            authority="representation_of_observed_sources",
            public_contract_ref=public_contract_ref,
            renderer=RENDERER,
            namespace=namespace,
            scope_quotes=list(scope_quotes),
            scope_authority="Read dependencies grant no edit permission;original caller scope stays authoritative.",
            files=files,
            relations=normalized,
        )
        text = dump(payload)
        tokens = count_text(text)
        if type(tokens) is not int or tokens < 0:
            raise ValueError("actual tokenizer returned invalid count")
        if tokens <= limit:
            versions = {s["path"]: s["sha"] for relation in chosen for s in relation["support"]}
            result = InterfaceMap(payload, text, tokens, versions)
            if result.current(current_version):
                return result
    return None
