"""Question-directed source views; model suggestions never become code facts.

No model calls, shell commands, candidate execution or acceptance live here.
Caller readers may use a container namespace; paths are never assumed to be
host paths. The local reader reuses the existing index and safe workbench.
"""

from dataclasses import dataclass, field
import copy
import json
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import uuid

from infinicoder.tools.code_workbench import CodeWorkbench
from infinicoder.memory.source.task_source_roots import relative_public_path
from infinicoder.execution.workspace_tools import Workspace


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SourcePlanInput:
    public_task: str
    task_identity: str
    namespace: dict
    remaining_calls: int
    remaining_seconds: float


@dataclass(frozen=True)
class PlanDraft:
    text: str
    queries: tuple = ()
    status: str = "unverified_model_hypothesis"
    issues: tuple = ()
    query_protocol: str = "none"


def _read_file_queries(text, workspace_aliases):
    """Accept only the model's literal read_file XML dialect, without execution."""
    if "<!" in text or "<?" in text:
        raise ValueError("XML_declarations_and_entities_are_not_read_queries")
    blocks = re.findall(r"<function\b[^>]*>.*?</function\s*>", text, re.S)
    if not 1 <= len(blocks) <= 4 or len(re.findall(r"<function\b", text)) != len(blocks):
        raise ValueError("one_to_four_complete_read_calls_required")
    queries = []
    for block in blocks:
        if len(block) > 8192:
            raise ValueError("bounded_read_call_required")
        try:
            node = ET.fromstring(block)
        except ET.ParseError as error:
            raise ValueError("invalid_read_XML") from error
        if (
            node.tag != "function"
            or node.attrib != {"name": "read_file"}
            or (node.text or "").strip()
        ):
            raise ValueError("only_read_file_is_supported")
        parameters = {}
        for child in node:
            if (
                child.tag != "param"
                or set(child.attrib) != {"name"}
                or list(child)
                or child.attrib["name"] in parameters
                or (child.tail or "").strip()
            ):
                raise ValueError("literal_unique_read_parameters_required")
            parameters[child.attrib["name"]] = (child.text or "").strip()
        paths = set(parameters) & {"file_path", "path"}
        if (
            set(parameters) - {"file_path", "path", "start_line", "end_line"}
            or len(paths) != 1
            or ("start_line" in parameters) != ("end_line" in parameters)
        ):
            raise ValueError("read_path_and_optional_complete_range_required")
        raw = parameters[next(iter(paths))]
        target = relative_public_path(raw, workspace_aliases)
        canonical = target is not None and (
            raw == target
            or any(raw == alias.rstrip("/") + "/" + target for alias in workspace_aliases)
        )
        if not canonical or not raw or len(raw) > 500 or "\0" in raw:
            raise ValueError("read_path_outside_explicit_workspace_or_not_normalized")
        query = dict(op="inspect", target=target, view="source")
        if "start_line" in parameters:
            if not all(
                re.fullmatch(r"[1-9][0-9]*", parameters[key]) for key in ("start_line", "end_line")
            ):
                raise ValueError("positive_integer_source_range_required")
            first, last = int(parameters["start_line"]), int(parameters["end_line"])
            if not first <= last < first + 80:
                raise ValueError("at_most_eighty_existing_lines")
            query.update(start=first, end=last)
        queries.append(query)
    return tuple(queries)


def parse_draft(text, *, complete=True, workspace_aliases=()):
    """Optional bounded JSON or read_file requests, never executable code."""
    if not isinstance(text, str):
        return PlanDraft("", status="invalid_response", issues=("non_text_response",))
    if not complete:
        return PlanDraft(text, status="incomplete_response;queries_not_executed")
    blocks = re.findall(r"```ic_source_queries\s*\n(.*?)\n```", text, re.S)
    if "<function" in text:
        if blocks:
            return PlanDraft(
                text, status="invalid_query_block", issues=("ambiguous_read_protocols",)
            )
        try:
            return PlanDraft(
                text, _read_file_queries(text, workspace_aliases), query_protocol="read_file_xml"
            )
        except (ValueError, TypeError) as error:
            return PlanDraft(text, status="invalid_query_block", issues=(str(error),))
    if not blocks:
        return PlanDraft(text, issues=("no_query_block",))
    if len(blocks) != 1:
        return PlanDraft(text, status="invalid_query_block", issues=("multiple_query_blocks",))
    try:
        items = json.loads(blocks[0])
        if not isinstance(items, list) or len(items) > 4:
            raise ValueError("at_most_four_queries")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("query_must_be_object")
            if item.get("op") == "locate":
                if (
                    set(item) != {"op", "query"}
                    or not isinstance(item["query"], str)
                    or not 1 <= len(item["query"]) <= 300
                ):
                    raise ValueError("bounded_literal_locate_required")
            elif item.get("op") == "inspect":
                if set(item) - {"op", "target", "start", "end", "view"} or not {
                    "op",
                    "target",
                } <= set(item):
                    raise ValueError("invalid_inspect_fields")
                target = item["target"]
                if not isinstance(target, str) or len(target) > 500:
                    raise ValueError("bounded_source_target_required")
                path = target.split("::", 1)[0]
                if relative_public_path(path) != path:
                    raise ValueError("normalized_relative_source_target_required")
                if ("start" in item) != ("end" in item):
                    raise ValueError("both_range_endpoints_required")
                if "start" in item and (
                    type(item["start"]) is not int
                    or type(item["end"]) is not int
                    or not 1 <= item["start"] <= item["end"] < item["start"] + 80
                ):
                    raise ValueError("at_most_eighty_existing_lines")
                if item.get("view", "source") not in ("source", "context", "relations", "flow"):
                    raise ValueError("unsupported_source_view")
            else:
                raise ValueError("unsupported_read_operation")
        return PlanDraft(text, tuple(copy.deepcopy(items)), query_protocol="ic_source_queries")
    except (ValueError, TypeError) as error:
        return PlanDraft(text, status="invalid_query_block", issues=(str(error),))


class WorkspaceSourcePlanReader:
    """Caller-selected host workspace, not an implicit Docker/host mapping."""

    identity = "workspace_source_plan_reader_v1"

    def __init__(self, request, index, *, count_output, remaining, model_root="/workspace"):
        self.index = index
        self.workspace_aliases = (model_root,)
        self.namespace = dict(
            environment="caller_host_workspace",
            root=str(Path(request.workspace).resolve()),
            model_root=model_root,
            source_coverage="queried_files_only",
        )
        self.workbench = CodeWorkbench(
            index,
            Workspace(request.workspace),
            count_output=count_output,
            remaining=remaining,
            output_limit=1800,
        )
        from infinicoder.memory.source.source_read_session import SourceReadSession

        self.session = SourceReadSession(
            index, self.workbench, workspace_aliases=self.workspace_aliases
        )
        self.index.db.execute(
            "CREATE TABLE IF NOT EXISTS source_plan_views(ref TEXT PRIMARY KEY,sha256 TEXT,record TEXT)"
        )
        self.index.db.commit()

    def seed(self, public_task):
        return self.session.seed(public_task)

    def query(self, item):
        view = self._read_view(item)
        return self._record_view(item, view)

    def _read_view(self, item):
        self.workbench.read_event_ref = "source-view:" + uuid.uuid4().hex
        try:
            return self.session.query(item)
        finally:
            self.workbench.read_event_ref = None

    def _record_view(self, item, view):
        # Append-only exact read returns in the existing task SQLite store.
        record = dump(dict(namespace=self.namespace, query=item, view=view))
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO source_plan_views VALUES(?,?,?)",
                (view["read_event_ref"], hashlib.sha256(record.encode()).hexdigest(), record),
            )
        return view

    def current_version(self, path):
        return self.workbench._read(path)[2]


class MappedWorkspaceSourcePlanReader(WorkspaceSourcePlanReader):
    """A caller-verified live bind; every used hash is checked in its namespace.

    The caller verifies the mount and supplies a fixed read-only version probe.
    No container command, ID or host mapping is accepted from model text.
    """

    identity = "mapped_workspace_source_plan_reader_v1;dual_namespace_versions"

    def __init__(self, request, index, *, mapping, namespace_version, count_output, remaining):
        if (
            not isinstance(mapping, dict)
            or not mapping.get("environment")
            or not mapping.get("binding_identity")
            or mapping.get("host_root") != str(Path(request.workspace).resolve())
            or not isinstance(mapping.get("model_root"), str)
            or not mapping["model_root"].startswith("/")
            or not callable(namespace_version)
        ):
            raise ValueError(
                "explicit caller-verified live source mapping and namespace version probe required"
            )
        super().__init__(
            request,
            index,
            count_output=count_output,
            remaining=remaining,
            model_root=mapping["model_root"],
        )
        self.namespace = dict(
            environment=mapping["environment"],
            root=mapping["model_root"],
            model_root=mapping["model_root"],
            host_mapped_root=mapping["host_root"],
            binding_identity=mapping["binding_identity"],
            source_coverage="queried_files_only",
            version_policy="safe host read plus actual namespace read must agree",
        )
        self.namespace_version = namespace_version

    def current_version(self, path):
        expected = super().current_version(path)
        observed = self.namespace_version(path)
        if expected != observed:
            raise ValueError("mapped source differs in the actual execution namespace")
        return observed

    def query(self, item):
        view = self._read_view(item)
        for path, expected in view.get("source_files", {}).items():
            # The read session already did its same-handle host checks and
            # post-read revalidation. One actual namespace probe per file.
            if self.namespace_version(path) != expected:
                raise ValueError("source changed between index/read/namespace observation")
        return self._record_view(item, view)


@dataclass
class SourcePlanPacket:
    namespace: dict
    draft: PlanDraft
    response_ref: str
    source_facts: list = field(default_factory=list)
    static_candidates: list = field(default_factory=list)
    unknown: list = field(default_factory=list)

    def add_view(self, query, view):
        """Only an actual caller read return can supply source/relationship data."""
        versions = view.get("source_files", {})
        if not isinstance(versions, dict) or any(
            not isinstance(p, str)
            or relative_public_path(p) != p
            or not isinstance(h, str)
            or re.fullmatch(r"[0-9a-f]{64}", h) is None
            for p, h in versions.items()
        ):
            raise ValueError("read return lacks normalized source versions")
        lines = {}
        for record in view.get("records", []):
            record = copy.deepcopy(record)
            if record.get("kind") == "source_line":
                path, digest = record.get("path"), record.get("source_sha256")
                if (
                    versions.get(path) != digest
                    or type(record.get("line")) is not int
                    or not isinstance(record.get("source"), str)
                ):
                    raise ValueError("source line lacks the actual read version")
                lines.setdefault((path, digest), []).append(
                    dict(line=record["line"], text=record["source"])
                )
            else:
                self.static_candidates.append(
                    dict(
                        query=query,
                        record=record,
                        support_sources=copy.deepcopy(versions),
                        read_event_ref=view.get("read_event_ref"),
                        status="static_or_text_candidate;runtime_dispatch_and_ownership_unknown",
                    )
                )
        for (path, digest), values in lines.items():
            for offset in range(0, len(values), 8):
                self.source_facts.append(
                    dict(
                        path=path,
                        source_sha256=digest,
                        lines=values[offset : offset + 8],
                        requested=query,
                        source_complete=False,
                        read_event_ref=view.get("read_event_ref"),
                        status="exact_read_excerpt;not_complete_function_or_correctness_proof",
                    )
                )
        if view.get("omitted_after") or view.get("next_cursor"):
            self.unknown.append(
                dict(
                    query=query,
                    status="read_view_has_omissions",
                    next_cursor=view.get("next_cursor"),
                )
            )

    def current(self, reader):
        """Recheck only observed files; rename/delete/failed read stays unknown."""
        versions = {}
        for fact in self.source_facts:
            versions[fact["path"]] = fact["source_sha256"]
        for candidate in self.static_candidates:
            versions.update(candidate["support_sources"])
        status = {}
        for path, expected in versions.items():
            try:
                status[path] = "current" if reader.current_version(path) == expected else "stale"
            except (OSError, ValueError, KeyError, UnicodeError):
                status[path] = "unknown_or_missing"
        facts = [copy.deepcopy(f) for f in self.source_facts if status.get(f["path"]) == "current"]
        candidates = [
            copy.deepcopy(c)
            for c in self.static_candidates
            if all(status.get(p) == "current" for p in c["support_sources"])
        ]
        invalid = [
            dict(path=p, status=s, old_sha256=versions[p])
            for p, s in status.items()
            if s != "current"
        ]
        return facts, candidates, invalid

    def bounded(self, reader, count_text, *, limit=1600, hypothesis_limit=400):
        from infinicoder.memory.source.source_packet_budget import bounded_prefix

        facts = copy.deepcopy(self.source_facts)
        candidates = copy.deepcopy(self.static_candidates)
        # Bytes only choose a hypothesis trial; admission measures the entire
        # real packet. Original unabridged planning text remains in the store.
        hypothesis = self.draft.text.encode()[:hypothesis_limit].decode("utf8", errors="ignore")
        value = dict(
            origin="caller_source_planning_v1;one_same_model_query",
            namespace=self.namespace,
            response_ref=self.response_ref,
            model_hypothesis=hypothesis,
            hypothesis_complete=hypothesis == self.draft.text,
            hypothesis_status=self.draft.status,
            source_facts=[],
            static_candidates=[],
            invalidated_sources=[],
            unknown=copy.deepcopy(self.unknown),
            omitted_source_records=len(facts),
            omitted_static_records=len(candidates),
            authority="Plan is a suggestion, not permission or validation. Read scope does not grant write scope. "
            "Only quoted current source is fact; static relations and runtime ownership remain uncertain. "
            "Earlier plan quotes become historical after source changes. Ordinary reads, edits and tests remain available.",
        )
        records = [("source_facts", record) for record in facts] + [
            ("static_candidates", record) for record in candidates
        ]

        def packet(selected, length):
            result = copy.deepcopy(value)
            for name, record in selected:
                result[name].append(record)
            result["omitted_source_records"] = len(facts) - len(result["source_facts"])
            result["omitted_static_records"] = len(candidates) - len(result["static_candidates"])
            return result

        value = bounded_prefix(records, packet, count_text, limit, count_limit=3)
        if value is None:
            return None
        # Only the actually selected source needs revalidation before input.
        selected = SourcePlanPacket(
            self.namespace,
            self.draft,
            self.response_ref,
            source_facts=value["source_facts"],
            static_candidates=value["static_candidates"],
        )
        current_facts, current_candidates, invalid = selected.current(reader)
        if invalid:
            value.update(
                source_facts=current_facts,
                static_candidates=current_candidates,
                invalidated_sources=invalid,
                omitted_source_records=len(facts) - len(current_facts),
                omitted_static_records=len(candidates) - len(current_candidates),
            )
            if count_text(dump(value)) > limit:
                return None
        return value
