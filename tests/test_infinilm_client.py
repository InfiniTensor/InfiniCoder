"""Client protocol and action-loop contracts without a model service."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import threading
import unittest
from unittest.mock import patch


@contextmanager
def service(replies):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append(
                (
                    self.path,
                    dict(self.headers),
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            status, reply = replies[min(len(calls) - 1, len(replies) - 1)]
            body = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", calls
        finally:
            server.shutdown()
            worker.join()


def completion(content="```mswea_bash_command\nprintf ready\n```", **choice):
    return dict(
        choices=[
            dict(message=dict(role="assistant", content=content), finish_reason="stop") | choice
        ],
        usage=dict(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


@unittest.skipUnless(
    importlib.util.find_spec("pydantic") and importlib.util.find_spec("jinja2"),
    "requires the text extra",
)
class InfiniLMTests(unittest.TestCase):
    def model(self, base):
        from infinicoder.models.infinilm_client import InfiniLMTextModel, default_config

        config = default_config()["model"]
        return InfiniLMTextModel(
            **(
                config
                | dict(
                    model_name="served/model",
                    model_kwargs=dict(
                        api_base=base,
                        api_key="owned-secret",
                        max_tokens=128,
                        temperature=0,
                        timeout=5,
                    ),
                )
            )
        )

    def test_text_request_usage_and_private_annotations(self):
        with service([(200, completion())]) as (base, calls):
            model = self.model(base)
            result = model.query(
                [dict(role="user", content="task", extra=dict(private="never send"))]
            )
            self.assertEqual(result["extra"]["actions"], [dict(command="printf ready")])
            self.assertEqual(result["extra"]["response"]["usage"]["total_tokens"], 18)
            self.assertEqual(result["extra"]["cost_status"], "unpriced")
            self.assertEqual(len(calls), 1)
            path, headers, body = calls[0]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(headers["Authorization"], "Bearer owned-secret")
            self.assertEqual(
                body,
                dict(
                    model="served/model",
                    messages=[dict(role="user", content="task")],
                    stream=False,
                    temperature=0,
                    max_tokens=128,
                ),
            )
            self.assertNotIn("owned-secret", json.dumps(model.serialize()))
            self.assertNotIn("owned-secret", json.dumps(model.get_template_vars()))
            self.assertNotIn("owned-secret", repr(model.config))

    def test_incomplete_or_ambiguous_actions_never_execute(self):
        from infinicoder.models.infinilm_client import Completion
        from infinicoder.models.model_client import ProviderError
        from infinicoder.runtime.native_protocol import FormatError

        for value in (
            dict(choices=[]),
            completion(message=dict(role="assistant", content=None)),
            completion(message=dict(role="assistant", content="", tool_calls=[dict(id="x")])),
        ):
            with self.assertRaises(ProviderError):
                Completion(value)
        for reply in (
            completion(finish_reason="length"),
            completion(content="no action"),
            completion(content="```mswea_bash_command\n\n```"),
        ):
            with service([(200, reply)]) as (base, calls):
                with self.assertRaises(FormatError) as caught:
                    self.model(base).query([dict(role="user", content="task")])
                self.assertEqual(caught.exception.messages[0]["extra"]["response"], reply)
                self.assertEqual(len(calls), 1)

        with service(
            [(400, dict(error=dict(code="context_length_exceeded", message="owned-secret")))]
        ) as (base, calls):
            with self.assertRaises(ProviderError) as caught:
                self.model(base).query([dict(role="user", content="task")])
            self.assertEqual(caught.exception.category, "context_overflow")
            self.assertNotIn("owned-secret", str(caught.exception))
            self.assertEqual(len(calls), 1)

    def test_http_failures_are_not_automatically_retried(self):
        from infinicoder.models.model_client import ProviderError

        for code in (401, 429, 500, 302):
            with service([(code, dict(error="owned-secret"))]) as (base, calls):
                with self.assertRaises(ProviderError) as caught:
                    self.model(base).query([dict(role="user", content="task")])
                self.assertEqual(caught.exception.status, code)
                self.assertNotIn("owned-secret", str(caught.exception))
                self.assertEqual(len(calls), 1)

    def test_only_confirmed_local_non_dispatch_can_retry(self):
        from infinicoder.models.infinilm_client import Completion
        from infinicoder.models.native_delivery_policy import ConfirmedNotDispatched

        model = self.model("http://127.0.0.1:1/v1")
        with patch.object(
            model, "_query", side_effect=[ConfirmedNotDispatched(), Completion(completion())]
        ) as query:
            model.query([dict(role="user", content="task")])
            self.assertEqual(query.call_count, 2)
        with patch.object(model, "_query", side_effect=ConfirmedNotDispatched()) as query:
            with self.assertRaises(ConfirmedNotDispatched):
                model.query([dict(role="user", content="task")])
            self.assertEqual(query.call_count, 2)

    def test_native_loop_executes_injected_environment_and_stops_on_submission(self):
        from infinicoder.models.infinilm_client import default_config
        from infinicoder.runtime.native_agent import NativeAgent
        from infinicoder.runtime.native_environment import NativeEnvironment

        class Environment(NativeEnvironment):
            def execute(self, action, **kwargs):
                executed.append(action["command"])
                output = dict(
                    output="ready"
                    if len(executed) == 1
                    else "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
                    returncode=0,
                    exception_info="",
                )
                self._check_finished(output)
                return output

        executed = []
        replies = [
            (200, completion()),
            (
                200,
                completion(
                    "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
                ),
            ),
        ]
        with service(replies) as (base, calls):
            agent = NativeAgent(
                self.model(base), Environment(), **(default_config()["agent"] | dict(step_limit=3))
            )
            outcome = agent.run("owned task")
            self.assertEqual(outcome["exit_status"], "Submitted")
            self.assertEqual(agent.n_calls, 2)
            self.assertEqual(
                executed, ["printf ready", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
            )
            self.assertNotIn("owned-secret", json.dumps(agent.serialize()))
        with self.assertRaises(NotImplementedError):
            NativeEnvironment().execute(dict(command="anything"))


if __name__ == "__main__":
    unittest.main()
