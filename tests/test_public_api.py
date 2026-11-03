"""Public configuration, CLI discovery and independent task acceptance."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from infinicoder.models.model_client import CompatibleClient, ProviderError

from infinicoder import Check, ModelConfig, ModelResponse, RuntimeConfig, TaskSpec, create_runtime
from infinicoder.app.cli import main


class PublicInterface(unittest.TestCase):
    def test_provider_credentials_stay_at_the_configured_endpoint(self):
        received = []

        class Target(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                received.append(dict(self.headers))
                self.send_response(200)
                self.end_headers()

            do_POST = do_GET

        class Origin(Target):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                if self.server.redirect_code:
                    self.send_response(self.server.redirect_code)
                    self.send_header("Location", f"http://127.0.0.1:{target.server_port}/sink")
                    self.end_headers()
                else:
                    data = dict(
                        choices=[dict(message=dict(content="ok"), finish_reason="stop")],
                        content=[dict(type="text", text="ok")],
                        stop_reason="end_turn",
                    )
                    body = json.dumps(data).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        with (
            ThreadingHTTPServer(("127.0.0.1", 0), Target) as target,
            ThreadingHTTPServer(("127.0.0.1", 0), Origin) as origin,
        ):
            threads = [
                threading.Thread(target=s.serve_forever, daemon=True) for s in (target, origin)
            ]
            for thread in threads:
                thread.start()
            try:
                for protocol in ("openai-compatible", "anthropic"):
                    client = CompatibleClient(
                        ModelConfig(
                            f"http://127.0.0.1:{origin.server_port}/v1",
                            "scripted",
                            api_key="example-only-secret",
                            protocol=protocol,
                        )
                    )
                    for code in (301, 302, 303, 307, 308):
                        origin.redirect_code = code
                        with self.subTest(protocol=protocol, status=code):
                            with self.assertRaises(ProviderError) as error:
                                client.generate([dict(role="user", content="example")], 3)
                            self.assertEqual(error.exception.category, "configuration")
                            self.assertFalse(error.exception.retryable)
                            self.assertNotIn("example-only-secret", str(error.exception))
                    origin.redirect_code = None
                    self.assertEqual(
                        client.generate([dict(role="user", content="example")], 3).text, "ok"
                    )
                self.assertEqual(received, [])
            finally:
                for server in (target, origin):
                    server.shutdown()
                for thread in threads:
                    thread.join()

    def test_configured_agent_edits_and_passes_independent_acceptance(self):
        original = "def answer():\n    return 0\n"
        source_hash = hashlib.sha256(original.encode()).hexdigest()
        actions = [
            dict(tool="read", arguments=dict(path="answer.py")),
            dict(
                tool="edit",
                arguments=dict(
                    path="answer.py", expected_hash=source_hash, old="return 0", new="return 42"
                ),
            ),
            dict(tool="finish", arguments={}),
        ]

        class Client:
            config = ModelConfig(
                "http://127.0.0.1:1/v1",
                "scripted",
                api_key="example-only-credential",
                max_output_tokens=256,
            )
            calls = 0

            def generate(self, messages, timeout):
                action = actions[self.calls]
                self.calls += 1
                return ModelResponse(
                    json.dumps(action),
                    usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "answer.py").write_bytes(original.encode("utf-8"))
            client = Client()
            runtime = create_runtime(
                workspace,
                client,
                root / "session",
                config=RuntimeConfig(
                    profile="general",
                    automatic_workers=False,
                    domain_compiler=False,
                    allow_commands=True,
                    max_model_calls=3,
                ),
            )
            check = Check(
                "acceptance",
                [sys.executable, "-c", "from answer import answer; assert answer() == 42"],
            )
            state = runtime.execute(
                TaskSpec("Make answer() return 42.", allowed_paths=["answer.py"], checks=[check])
            )
            self.assertEqual(state["status"], "succeeded", state["reason"])
            self.assertEqual(client.calls, 3)
            self.assertIn("return 42", (workspace / "answer.py").read_text())
            self.assertNotIn("example-only-credential", repr(client.config))
            self.assertNotIn("example-only-credential", (root / "session/state.json").read_text())
            client.config = replace(client.config, api_key="rotated-example-credential")
            resumed = runtime.execute(
                TaskSpec("Make answer() return 42.", allowed_paths=["answer.py"], checks=[check]),
                resume=True,
            )
            self.assertEqual(resumed["status"], "succeeded")
            changed = create_runtime(
                workspace,
                client,
                root / "session",
                config=RuntimeConfig(
                    profile="general",
                    automatic_workers=False,
                    domain_compiler=False,
                    allow_commands=True,
                    max_model_calls=4,
                ),
            )
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                changed.execute(
                    TaskSpec(
                        "Make answer() return 42.", allowed_paths=["answer.py"], checks=[check]
                    ),
                    resume=True,
                )
            self.assertEqual(client.calls, 3)

    def test_common_help_and_advanced_discovery(self):
        common, advanced = io.StringIO(), io.StringIO()
        with redirect_stdout(common), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)
        with redirect_stdout(advanced):
            self.assertEqual(main(["--advanced-help"]), 0)
        self.assertIn("--resume", common.getvalue())
        self.assertNotIn("--implementation-provider", common.getvalue())
        self.assertIn("--implementation-provider", advanced.getvalue())

    def test_invalid_invocation_has_machine_readable_error(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = main(
                [
                    "--workspace",
                    ".",
                    "--spec",
                    "absent-task.json",
                    "--state-dir",
                    "unused-state",
                    "--base-url",
                    "http://127.0.0.1:1/v1",
                    "--model",
                    "unused",
                ]
            )
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
