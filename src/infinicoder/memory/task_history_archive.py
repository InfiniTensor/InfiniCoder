"""Task-private read-only export of exact history/source records; no replay."""

import hashlib
import json
from pathlib import Path

MODEL_ROOT = "/tmp/ic-task-history"

# This stdlib-only reader runs as an ordinary caller-owned shell action. It has
# no SQLite handle, model client, executor, project import or private checker.
READ_PROGRAM = """import hashlib,json,re,sys
from pathlib import Path
root=Path(__file__).resolve().parent
identity=json.loads((root/'identity.json').read_bytes()) if (root/'identity.json').is_file() else {}
def dump(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def load(kind,ref):
 if kind=='event':
  if not re.fullmatch(r'[0-9]+',ref) or int(ref)<2:raise ValueError('event id must be >=2')
  name=str(int(ref))
 elif kind=='source':
  if not re.fullmatch(r'source-view:[0-9a-f]{32}',ref):raise ValueError('copy an exact source-view ref')
  name=ref.replace(':','_')
 elif kind in ('hypothesis','observation'):
  if not re.fullmatch(r'[0-9a-f]{64}',ref):raise ValueError('copy an exact hypothesis ref')
  name=ref
 else:raise ValueError('expected event/source/hypothesis/observation')
 path=root/({'event':'events','source':'sources','hypothesis':'hypotheses','observation':'observations'}[kind])/(name+'.json')
 if path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2:raise ValueError('record unavailable or too large')
 record=json.loads(path.read_bytes());text=dump(record['payload'])
 if any(record.get(key)!=value for key,value in identity.items()):raise ValueError('archive phase/namespace differs')
 if hashlib.sha256(text.encode()).hexdigest()!=record['sha256']:raise ValueError('archive record hash differs')
 return record,text
try:
 args=sys.argv[1:]
 if args and args[0]=='search' and len(args)==2 and 0<len(args[1])<=300:
  found=[]
  paths=sorted((root/'events').glob('*.json'),key=lambda p:int(p.stem),reverse=True)
  for path in paths:
   record,text=load('event',path.stem)
   if args[1].lower() in text.lower():
    found.append(dict(id=int(path.stem),sha256=record['sha256'],preview=text[:240],historical=True,**identity))
    if len(found)==8:break
  print(dump(dict(view='historical_search',records=found,limited_to=8,commands_replayed=False)))
 elif len(args) in (2,4):
  kind,ref=args[:2];start,end=(int(args[2]),int(args[3])) if len(args)==4 else (0,4000)
  if not 0<=start<end<=start+4000:raise ValueError('explicit character range at most4000 required')
  record,text=load(kind,ref)
  print(dump(dict(view='historical_'+kind,ref=ref,sha256=record['sha256'],start=start,end=min(end,len(text)),
   text=text[start:end],characters=len(text),more=end<len(text),current_source_proof=False,commands_replayed=False,**identity)))
 else:raise ValueError('usage: read.py event ID [START END] | source REF [START END] | hypothesis REF [START END] | observation REF [START END] | search TEXT')
except (ValueError,OSError,KeyError) as error:
 print(dump(dict(error=type(error).__name__+': '+str(error),commands_replayed=False)));raise SystemExit(2)
"""


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TaskHistoryArchive:
    """Append only known task tables, never expose the runtime's state folder."""

    def __init__(self, db, directory, *, remaining, phase=None, namespace=None):
        self.db, self.directory, self.remaining = db, Path(directory), remaining
        if (phase is None) != (namespace is None):
            raise ValueError("phase and namespace must be supplied together")
        if phase is not None and (
            phase not in ("plain", "A", "B")
            or not isinstance(namespace, str)
            or len(namespace) != 64
            or any(c not in "0123456789abcdef" for c in namespace)
        ):
            raise ValueError("identified phase history namespace required")
        self.identity = {} if phase is None else dict(phase=phase, namespace=namespace)
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            raise ValueError("history export cannot have linked components")
        self.directory.mkdir(mode=0o755, parents=False, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError("fresh empty history export required")
        for name in ("events", "sources", "hypotheses", "observations"):
            (self.directory / name).mkdir(mode=0o755)
        reader = self.directory / "read.py"
        reader.write_bytes(READ_PROGRAM.encode())
        reader.chmod(0o444)
        if self.identity:
            identity = self.directory / "identity.json"
            identity.write_bytes(dump(self.identity).encode())
            identity.chmod(0o444)
        self.exported = set()
        self.cursors = dict(events=1, sources=0, hypotheses=0, observations=0)

    def _write(self, kind, ref, payload, expected):
        key = (kind, ref)
        if key in self.exported:
            return
        self.remaining()
        body = dump(payload)
        if hashlib.sha256(body.encode()).hexdigest() != expected:
            raise ValueError("canonical task record hash differs")
        raw = dump(
            dict(kind=kind, ref=ref, sha256=expected, payload=payload, **self.identity)
        ).encode()
        if len(raw) > 16 * 1024**2:
            raise ValueError("history record exceeds retrievable16MiB limit")
        name = str(ref).replace(":", "_") + ".json"
        if "/" in name or "\\" in name:
            raise ValueError("normalized archive record name required")
        target = self.directory / kind / name
        temporary = target.with_suffix(".pending")
        with temporary.open("xb") as stream:
            stream.write(raw)
        temporary.chmod(0o444)
        temporary.replace(target)
        self.exported.add(key)

    def sync(self):
        for row in self.db.execute(
            "SELECT id,sha,message FROM native_history WHERE id>? ORDER BY id",
            (self.cursors["events"],),
        ):
            self._write("events", str(row["id"]), json.loads(row["message"]), row["sha"])
            self.cursors["events"] = row["id"]
        tables = {
            r[0] for r in self.db.execute('SELECT name FROM sqlite_master WHERE type="table"')
        }
        if "source_plan_views" in tables:
            for row in self.db.execute(
                "SELECT rowid,ref,sha256,record FROM source_plan_views WHERE rowid>? ORDER BY rowid",
                (self.cursors["sources"],),
            ):
                self._write("sources", row["ref"], json.loads(row["record"]), row["sha256"])
                self.cursors["sources"] = row["rowid"]
        if "task_evidence_hypotheses" in tables:
            for row in self.db.execute(
                "SELECT rowid,ref,record FROM task_evidence_hypotheses WHERE rowid>? ORDER BY rowid",
                (self.cursors["hypotheses"],),
            ):
                self._write("hypotheses", row["ref"], json.loads(row["record"]), row["ref"])
                self.cursors["hypotheses"] = row["rowid"]
        if "task_evidence_observations" in tables:
            for row in self.db.execute(
                "SELECT rowid,ref,record FROM task_evidence_observations WHERE rowid>? ORDER BY rowid",
                (self.cursors["observations"],),
            ):
                self._write("observations", row["ref"], json.loads(row["record"]), row["ref"])
                self.cursors["observations"] = row["rowid"]
