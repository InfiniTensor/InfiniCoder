"""Versioned module routing and separately stored, verified task experiences."""

import hashlib
import json
from pathlib import PurePosixPath
import sqlite3
from infinicoder.analysis.code_context import words


class HierarchicalMemory:
    def __init__(self, index):
        self.index = index

    def route(self, query, limit=8):
        from infinicoder.memory.repository_memory import RepositoryMemory

        return RepositoryMemory(self.index).route(query, limit)

    def expand(self, module=".", symbol_id=None, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        if symbol_id:
            symbol = self.index.symbol(symbol_id)
            file = self.index.db.execute(
                "SELECT hash FROM files WHERE path=?", (symbol["path"],)
            ).fetchone()
            return {
                "revision": self.index.revision(),
                "level": "symbol",
                "source_sha256": file["hash"],
                "symbol": symbol,
                "relationships": self.index.relationships(symbol_id, limit=limit),
                "structure": self.index.structure(symbol_id),
                "next": "Read the named source range;use_def(path,name) expands local data dependencies. Static candidates need source verification.",
            }
        p = PurePosixPath(module)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("unsafe module")
        prefix = "" if module == "." else str(p).rstrip("/") + "/"
        paths = [f["path"] for f in self.index.files() if f["path"].startswith(prefix)]
        symbols = []
        for path in paths:
            for row in self.index.db.execute(
                "SELECT s.id,s.path,s.signature,s.line,s.end,s.confidence,f.hash AS source_sha256 FROM symbols s JOIN files f ON f.path=s.path WHERE s.path=? ORDER BY s.line",
                (path,),
            ):
                if len(symbols) < limit:
                    symbols.append(dict(row))
        versions = {
            row["path"]: row["hash"] for row in self.index.files() if row["path"] in paths[:limit]
        }
        result = {
            "revision": self.index.revision(),
            "level": "module",
            "module": module,
            "files": paths[:limit],
            "omitted_files": max(0, len(paths) - limit),
            "symbols": symbols,
            "source_versions": versions,
            "next": "expand a symbol, then read its exact versioned source",
        }
        while len(json.dumps(result, ensure_ascii=False)) > 4500:
            if result["symbols"]:
                result["symbols"].pop()
            elif result["files"]:
                removed = result["files"].pop()
                result["source_versions"].pop(removed, None)
                result["omitted_files"] += 1
            else:
                break
        result["omitted_symbols"] = max(
            0,
            sum(
                1
                for path in paths
                for _ in self.index.db.execute("SELECT 1 FROM symbols WHERE path=?", (path,))
            )
            - len(result["symbols"]),
        )
        return result


class ExperienceBank:
    """Opt-in caller-owned bank, never a global hidden cross-benchmark cache."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS experiences(id TEXT PRIMARY KEY, task TEXT, insight TEXT, conditions TEXT, proof TEXT)"
        )

    def record(self, task, insight, conditions, proof):
        if (
            not isinstance(proof, dict)
            or proof.get("accepted") is not True
            or not proof.get("checks_run", 0) > 0
        ):
            raise ValueError("independent accepted checks required")
        if not isinstance(proof.get("source_sha256"), str) or len(proof["source_sha256"]) != 64:
            raise ValueError("verified source identity required")
        if (
            not isinstance(conditions, dict)
            or not isinstance(insight, str)
            or not insight.strip()
            or len(insight) > 2000
        ):
            raise ValueError("bounded insight and explicit conditions required")
        identity = hashlib.sha256(
            json.dumps([task, insight, conditions, proof], sort_keys=True).encode()
        ).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO experiences VALUES(?,?,?,?,?)",
                (identity, task, insight, json.dumps(conditions), json.dumps(proof)),
            )
        return identity

    def retrieve(self, query, facts, limit=3):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("limit must be 1..20")
        hits = []
        for identity, task, insight, conditions, proof in self.db.execute(
            "SELECT * FROM experiences"
        ):
            conditions = json.loads(conditions)
            if any(facts.get(k) != v for k, v in conditions.items()):
                continue
            score = len(set(words(query)) & set(words(task)))
            if score:
                hits.append(
                    dict(
                        id=identity,
                        insight=insight,
                        conditions=conditions,
                        proof=json.loads(proof),
                        score=score,
                    )
                )
        return sorted(hits, key=lambda x: (-x["score"], x["id"]))[:limit]

    def close(self):
        self.db.close()
