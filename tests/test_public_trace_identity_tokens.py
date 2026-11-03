import json, tempfile, unittest
from pathlib import Path
from infinicoder.verification.public_replay import PublicTraceRecorder


class TraceIdentityTests(unittest.TestCase):
    def test_distinct_process_objects_have_stable_trace_local_alias_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "owned.py"
            source.write_bytes(b"def observe(a,b):\n    return [a,b,a]\n")
            ns = {}
            exec(compile(source.read_bytes(), str(source.resolve()), "exec"), ns)

            class Hostile:
                def __repr__(self):
                    raise AssertionError("no user repr")

            objects = [Hostile() for _ in range(4)]
            events = []
            for a, b in ((objects[0], objects[1]), (objects[2], objects[3])):
                recorder = PublicTraceRecorder(root, ("owned.py",))
                with recorder:
                    ns["observe"](a, b)
                events.append(recorder.report()["events"])
            self.assertEqual(events[0], events[1])
            a = events[0][0]["parameters"]["a"]["observed_object_id"]
            b = events[0][0]["parameters"]["b"]["observed_object_id"]
            self.assertEqual((a, b), (1, 2))
            returned = events[0][-1]["returned"]["items"]
            self.assertEqual([x["observed_object_id"] for x in returned], [a, b, a])
            self.assertFalse(any(str(id(o)) in json.dumps(events) for o in objects))


if __name__ == "__main__":
    unittest.main()
