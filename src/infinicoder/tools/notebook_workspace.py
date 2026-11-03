"""Versioned notebook cell sources;no kernel execution or saved-output claims."""

import ast
import hashlib
import json
from pathlib import Path

GUIDANCE = "notebook_cells {path,start?:0,count?:20};notebook_read {path,cell,start?:1,count?:100};notebook_edit {path,cell,expected_hash,expected_cell_hash,old,new}. Notebook tools use cell indexes and cell-local lines,not JSON file lines. notebook_cells lists source cells;notebook_read reads one cell. notebook_edit replaces one unique source substring with both current file and cell hashes. Other cells/metadata are preserved;all saved code outputs are cleared after an edit. Saved outputs never prove current correctness. Use the original public checks and final acceptance."


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


class NotebookWorkspace:
    def __init__(self, workspace, *, count_tokens=None, token_limit=1600, protected_paths=()):
        if type(token_limit) is not int or not 256 <= token_limit <= 4000:
            raise ValueError("notebook token_limit must be256..4000")
        if count_tokens is not None and not callable(count_tokens):
            raise ValueError("count_tokens must be callable")
        self.workspace = workspace
        self.count = count_tokens or (lambda text: len(text.encode("utf-8")))
        self.token_limit = token_limit
        self.count_kind = "caller_counter" if count_tokens else "UTF8_byte_bound"
        self.protected = tuple(
            (Path(p) if Path(p).is_absolute() else workspace.root / p).resolve()
            for p in protected_paths
        )

    def _load(self, path, write=False):
        file = self.workspace.path(path, write=write)
        if file.suffix.lower() != ".ipynb":
            raise ValueError("notebook tools require an ipynb file")
        if any(file == p or file.is_relative_to(p) for p in self.protected):
            raise ValueError("protected notebook")
        if not file.is_file():
            raise ValueError("regular notebook file required")
        if file.stat().st_size > 2_000_000:
            raise ValueError("notebook too large")
        raw = file.read_bytes()

        def invalid(value):
            raise ValueError("nonfinite notebook JSON")

        document = json.loads(raw.decode("utf-8-sig"), parse_constant=invalid)
        if (
            not isinstance(document, dict)
            or type(document.get("nbformat")) is not int
            or document["nbformat"] != 4
            or not isinstance(document.get("cells"), list)
            or len(document["cells"]) > 10000
        ):
            raise ValueError("nbformat4 notebook with at most10000 cells required")
        if not isinstance(document.get("metadata", {}), dict):
            raise ValueError("invalid notebook metadata")
        return raw, document

    @staticmethod
    def _source(document, cell):
        if type(cell) is not int or not 0 <= cell < len(document["cells"]):
            raise ValueError("notebook cell index out of range")
        entry = document["cells"][cell]
        if not isinstance(entry, dict) or entry.get("cell_type") not in ("code", "markdown", "raw"):
            raise ValueError("invalid notebook cell")
        source = entry.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("notebook source must be text or text lines")
        return entry, source

    def cells(self, path, start=0, count=20):
        if type(start) is not int or start < 0 or type(count) is not int or not 1 <= count <= 32:
            raise ValueError("invalid notebook cell page")
        raw, document = self._load(path)
        total = len(document["cells"])
        if start > total:
            raise ValueError("notebook page starts beyond cell count")
        value = dict(
            path=path,
            sha256=sha(raw),
            nbformat=4,
            total_cells=total,
            start=start,
            cells=[],
            coordinate_kind="cell_local_lines",
            counter=self.count_kind,
        )
        for i in range(start, min(total, start + count)):
            entry, source = self._source(document, i)
            row = dict(
                cell=i,
                cell_type=entry["cell_type"],
                cell_sha256=sha(source.encode("utf-8")),
                lines=len(source.splitlines()),
            )
            if entry["cell_type"] == "code":
                row["declarations"] = []
                try:
                    if len(source) > 64000:
                        raise ValueError("cell analysis size limit")
                    tree = ast.parse(source)
                    for node in tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            row["declarations"].append(
                                dict(
                                    name=node.name[:200],
                                    kind="class" if isinstance(node, ast.ClassDef) else "function",
                                    line=node.lineno,
                                    end=node.end_lineno,
                                )
                            )
                            if len(row["declarations"]) == 6:
                                break
                    row["parse_status"] = "python_syntax_only"
                except (SyntaxError, ValueError, RecursionError):
                    row["parse_status"] = "unknown_or_non_python_cell"
            proposal = dict(
                value, cells=[*value["cells"], row], next_cell=i + 1 if i + 1 < total else None
            )
            if self.count(encoded(proposal)) > self.token_limit:
                break
            value = proposal
        value.setdefault("next_cell", start if start < total else None)
        if self.count(encoded(value)) > self.token_limit:
            raise ValueError("notebook metadata exceeds output budget")
        return value

    def read(self, path, cell, start=1, count=100):
        if type(start) is not int or start < 1 or type(count) is not int or not 1 <= count <= 120:
            raise ValueError("invalid notebook source range")
        raw, document = self._load(path)
        entry, source = self._source(document, cell)
        lines = source.splitlines()
        if start > max(1, len(lines)):
            raise ValueError("notebook source range starts beyond cell")
        value = dict(
            path=path,
            sha256=sha(raw),
            cell=cell,
            cell_type=entry["cell_type"],
            cell_sha256=sha(source.encode("utf-8")),
            coordinate_kind="cell_local_lines",
            total_lines=len(lines),
            start=start,
            text="",
            covered_line_count=0,
            source_complete=False,
            counter=self.count_kind,
            next_read_line=start if lines else None,
        )
        for line in lines[start - 1 : start - 1 + count]:
            n = value["covered_line_count"] + 1
            text = value["text"] + ("\n" if n > 1 else "") + line
            proposal = dict(
                value,
                text=text,
                covered_line_count=n,
                source_complete=start == 1 and n == len(lines),
                next_read_line=start + n if start + n <= len(lines) else None,
            )
            if self.count(encoded(proposal)) > self.token_limit:
                break
            value = proposal
        value["next_read_line"] = (
            start + value["covered_line_count"]
            if start + value["covered_line_count"] <= len(lines)
            else None
        )
        if self.count(encoded(value)) > self.token_limit:
            raise ValueError("notebook metadata exceeds output budget")
        return value

    def edit_arguments(self, path, cell, expected_hash, expected_cell_hash, old, new):
        raw, document = self._load(path, write=True)
        if sha(raw) != expected_hash:
            raise ValueError("notebook file changed;read again")
        entry, source = self._source(document, cell)
        if sha(source.encode("utf-8")) != expected_cell_hash:
            raise ValueError("notebook cell changed;read again")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("old/new source text required")
        if len(old) > 32768 or len(new) > 32768:
            raise ValueError("notebook patch too large")
        editable = source
        if editable.count(old) != 1:
            editable = source.replace("\r\n", "\n")
            old = old.replace("\r\n", "\n")
            new = new.replace("\r\n", "\n")
        if editable.count(old) != 1:
            raise ValueError("old text must match exactly once in the selected cell")
        changed = editable.replace(old, new, 1)
        if changed == source:
            raise ValueError("notebook patch leaves source unchanged")
        entry["source"] = (
            changed.splitlines(keepends=True) if isinstance(entry["source"], list) else changed
        )
        cleared = 0
        for other in document["cells"]:
            if isinstance(other, dict) and other.get("cell_type") == "code":
                cleared += bool(other.get("outputs")) or other.get("execution_count") is not None
                other.update(outputs=[], execution_count=None)
        result = json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
        return dict(
            path=path, expected_hash=expected_hash, old=raw.decode("utf-8-sig"), new=result
        ), dict(
            cell=cell,
            cell_sha256=sha(changed.encode("utf-8")),
            cleared_code_outputs=cleared,
            coordinate_kind="cell_local_lines",
        )

    def edit(self, **arguments):
        edit, receipt = self.edit_arguments(**arguments)
        return dict(self.workspace.edit(**edit), notebook=receipt)
