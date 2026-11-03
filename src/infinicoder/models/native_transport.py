"""Budgeted compatible-endpoint transport for the packaged Native CPU entry.

One loopback relay counts physical attempts,including SDK retries. It does not
rewrite prompts or implement an agent loop. Provider credentials stay here.
"""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from infinicoder.runtime.agent_execution import ModelBinding, UnsupportedCore
from infinicoder.models.model_client import validate_model_config


class LocalChatCounter:
    """Frozen local tokenizer,NoThink chat template;no weights or generation."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock = threading.RLock()
        self.tokenizer = None
        names = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "spiece.model",
            "config.json",
        )
        self.files = {
            name: hashlib.sha256((self.path / name).read_bytes()).hexdigest()
            for name in names
            if (self.path / name).is_file()
        }
        if not self.files:
            raise ValueError("existing local tokenizer directory required")
        self.identity = (
            "local_chat_NoThink:"
            + hashlib.sha256(json.dumps(self.files, sort_keys=True).encode()).hexdigest()
        )

    def __call__(self, messages, remaining):
        started = time.monotonic()
        if remaining <= 0 or not self.lock.acquire(timeout=max(0, remaining)):
            raise TimeoutError("original tokenizer deadline expired")
        try:
            if self.tokenizer is None:
                from transformers import AutoTokenizer

                if any(
                    hashlib.sha256((self.path / name).read_bytes()).hexdigest() != digest
                    for name, digest in self.files.items()
                ):
                    raise ValueError("local tokenizer changed before loading")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.path, local_files_only=True, trust_remote_code=False
                )
            value = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            if time.monotonic() - started >= remaining:
                raise TimeoutError("original tokenizer deadline expired")
            return len(value["input_ids"] if hasattr(value, "get") else value)
        finally:
            self.lock.release()


class NativeCompatibleTransport:
    identity = "compatible_physical_HTTP_budget_v1;local_actual_tokenizer;no_prompt_rewrite"

    def __init__(self, config, counter, *, delivery_policy=None):
        from infinicoder.models.native_delivery_policy import DeliveryPolicy

        if delivery_policy is not None and not isinstance(delivery_policy, DeliveryPolicy):
            raise TypeError("typed delivery policy required")
        self.config = config
        self.counter = counter
        self.delivery_policy = delivery_policy
        if delivery_policy is not None:
            self.identity += ";logical_query_delivery_v1"

    def request_observations(self):
        """Read-only caller-thread view of existing physical receipt states."""
        binding = getattr(self, "_request_observations", None)
        return binding[0]() if binding is not None and binding[1] == threading.get_ident() else None

    @contextmanager
    def bind_model(self, model, *, task_expired=None):
        if self.delivery_policy is None:
            yield
            return
        controller = getattr(self, "_delivery", None)
        if controller is None:
            raise UnsupportedCore("delivery model binding requires the active transport")
        with controller.bind_model(model, task_expired=task_expired):
            yield

    @contextmanager
    def limit_request(self, *, output_tokens, absolute_deadline, max_physical_calls=None):
        """A trusted sequential phase shares the same physical-request ledger."""
        if type(output_tokens) is not int or output_tokens < 1:
            raise ValueError("positive output token cap required")
        import math

        if (
            isinstance(absolute_deadline, bool)
            or not isinstance(absolute_deadline, (int, float))
            or not math.isfinite(absolute_deadline)
        ):
            raise ValueError("finite absolute phase deadline required")
        if max_physical_calls is not None and (
            type(max_physical_calls) is not int or max_physical_calls < 1
        ):
            raise ValueError("positive physical phase call bound required")
        binding = getattr(self, "_limits_binding", None)
        if binding is None or binding[2] != threading.get_ident():
            raise UnsupportedCore("request limits require the active caller transport thread")
        lock, phase, _ = binding
        with lock:
            if phase[0] is not None:
                raise UnsupportedCore("request limits cannot nest")
            phase[0] = dict(output_tokens=output_tokens, deadline=absolute_deadline)
            if max_physical_calls is not None:
                phase[0].update(max_physical_calls=max_physical_calls, attempts=0)
        try:
            yield
        finally:
            with lock:
                phase[0] = None

    def remaining_physical_calls(self):
        binding = getattr(self, "_physical_budget", None)
        if binding is None or binding[1] != threading.get_ident():
            raise UnsupportedCore("physical budget requires the active caller transport thread")
        return binding[0]()

    def validate(self, request):
        validate_model_config(self.config)
        url = urllib.parse.urlsplit(self.config.base_url)
        if (
            self.config.protocol != "openai-compatible"
            or url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise UnsupportedCore(
                "explicit compatible HTTP base URL without embedded credentials required"
            )
        if self.config.reasoning_prefix_open or self.config.structured_outputs:
            raise UnsupportedCore(
                "Native text protocol does not use Runtime structured/prefill modes"
            )
        allowed = {
            "seed",
            "top_p",
            "min_p",
            "repetition_penalty",
            "skip_special_tokens",
            "chat_template_kwargs",
        }
        if set(self.config.extra_body) - allowed:
            raise UnsupportedCore(
                "Native CPU transport accepts sampling fields only;use a trusted adapter for other protocols"
            )
        template = self.config.extra_body.get("chat_template_kwargs", {"enable_thinking": False})
        if (
            not isinstance(template, dict)
            or set(template) != {"enable_thinking"}
            or template["enable_thinking"] is not False
        ):
            raise UnsupportedCore("local CPU counter and transport require enable_thinking=False")
        if not callable(self.counter) or not getattr(self.counter, "identity", None):
            raise UnsupportedCore("actual identified chat-tokenizer callback required")
        if (
            self.delivery_policy is not None
            and self.delivery_policy.max_wait_seconds > request.budget.request_timeout
        ):
            raise UnsupportedCore(
                "logical wait exceeds the caller request timeout;declare compatible waiting layers before generation"
            )

    @contextmanager
    def open(self, request, deadline):
        self.validate(request)
        config = self.config
        counter = self.counter
        state = Path(request.state_dir)
        folder = state / "native-http"
        folder.mkdir(mode=0o700, exist_ok=False)
        ledger = state / "native-api.json"
        lock = threading.RLock()
        rows = []
        closing = False
        phase = [None]
        if getattr(self, "_limits_binding", None) is not None:
            raise UnsupportedCore("transport already bound")
        self._limits_binding = (lock, phase, threading.get_ident())

        def remaining_calls():
            with lock:
                return max(0, request.budget.max_model_calls - len(rows))

        self._physical_budget = (remaining_calls, threading.get_ident())

        def observations():
            with lock:
                return tuple(
                    {name: row.get(name) for name in ("attempt", "dispatch_attempted", "status")}
                    for row in rows
                )

        self._request_observations = (observations, threading.get_ident())
        client_attempts = 0
        rejected_without_dispatch = 0
        delivery = None
        token = secrets.token_hex(32)
        started = time.monotonic()
        base = config.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")

        # Keep credentials at the chosen endpoint;redirects are not a new endpoint authorization.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

        def save():
            with lock:
                value = dict(
                    model=config.model,
                    tokenizer_identity=counter.identity,
                    tokenizer_files=getattr(counter, "files", None),
                    physical_call_limit=request.budget.max_model_calls,
                    request_timeout_s=request.budget.request_timeout,
                    deadline_monotonic=deadline,
                    calls=rows,
                    client_attempts=client_attempts,
                    rejected_without_dispatch=rejected_without_dispatch,
                    closed=closing,
                    usage_complete=all(
                        not r.get("dispatch_attempted") or isinstance(r.get("usage"), dict)
                        for r in rows
                    ),
                    elapsed_s=time.monotonic() - started,
                )
                if delivery is not None:
                    value["logical_delivery"] = delivery.snapshot()
                temporary = ledger.with_suffix(".next")
                temporary.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8"
                )
                temporary.replace(ledger)

        if self.delivery_policy is not None:
            from infinicoder.models.native_delivery_policy import QueryDelivery

            delivery = QueryDelivery(
                self.delivery_policy,
                deadline,
                lock=lock,
                save=save,
                phase_deadline=lambda: phase[0]["deadline"] if phase[0] is not None else deadline,
            )
        self._delivery = delivery

        def dump(name, raw):
            path = folder / name
            with path.open("xb") as stream:
                stream.write(raw)
            return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, value):
                raw = value if isinstance(value, bytes) else json.dumps(value).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (OSError, ValueError):
                    pass

            def error(self, code, category):
                error = dict(type="invalid_request_error", message=category)
                if code == 400 and category.startswith("context_length_exceeded"):
                    # The SDK already aborts ContextWindowExceededError. The
                    # compatible error code and message must preserve that
                    # classification instead of retrying an unchanged input.
                    error.update(
                        code="context_length_exceeded",
                        message="This model's maximum context length is exceeded;" + category,
                    )
                self.reply(code, dict(error=error))

            def do_POST(self):
                nonlocal client_attempts, rejected_without_dispatch
                if self.path != "/v1/chat/completions":
                    return self.error(404, "unsupported route")
                if self.headers.get("Authorization") != "Bearer " + token:
                    return self.error(401, "local transport binding required")
                tick = time.monotonic()
                lease = None
                with lock:
                    client_attempts += 1
                    if delivery is not None:
                        lease = delivery.lookup(self.headers.get("X-InfiniCoder-Query-Id"))
                        if not delivery.accept(lease):
                            rejected_without_dispatch += 1
                            save()
                            return self.error(
                                400, "logical query inactive or terminal;no upstream dispatch"
                            )
                    if (
                        phase[0] is not None
                        and "max_physical_calls" in phase[0]
                        and phase[0]["attempts"] >= phase[0]["max_physical_calls"]
                    ):
                        rejected_without_dispatch += 1
                        save()
                        return self.error(400, "physical phase request budget exhausted")
                    if phase[0] is not None and "max_physical_calls" in phase[0]:
                        phase[0]["attempts"] += 1
                    limits = dict(phase[0]) if phase[0] is not None else None
                    if closing or tick >= deadline or len(rows) >= request.budget.max_model_calls:
                        rejected_without_dispatch += 1
                        save()
                        return self.error(400, "physical request or task budget exhausted")
                    row = dict(
                        attempt=len(rows) + 1,
                        started_monotonic=tick,
                        dispatch_attempted=False,
                        usage=None,
                    )
                    rows.append(row)
                    save()

                def update(**fields):
                    with lock:
                        row.update(fields)

                cutoff = min(
                    deadline,
                    tick + request.budget.request_timeout,
                    limits["deadline"] if limits else deadline,
                    lease["deadline_monotonic"] if lease is not None else deadline,
                )
                if lease is not None:
                    update(
                        logical_query_id=lease["query_id"],
                        logical_deadline=lease["deadline_monotonic"],
                    )
                update(phase_limits=limits)

                def remaining():
                    left = cutoff - time.monotonic()
                    if left <= 0 or closing:
                        raise TimeoutError("shared task/request deadline expired")
                    return left

                try:
                    self.connection.settimeout(remaining())
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 8 * 1024**2:
                        raise ValueError("bounded request required")
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError("incomplete request")
                    body = json.loads(raw)
                    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                        raise ValueError("text chat request required")
                    if body.get("n", 1) != 1 or body.get("stream", False):
                        raise ValueError("one nonstreaming completion required")
                    body.update(
                        model=config.model,
                        max_tokens=min(
                            request.model_profile.output_reserve,
                            limits["output_tokens"]
                            if limits
                            else request.model_profile.output_reserve,
                        ),
                        stream=False,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    if config.temperature is not None:
                        body["temperature"] = config.temperature
                    body.update(config.extra_body)
                    for name in ("max_completion_tokens", "stream_options"):
                        body.pop(name, None)
                    blob = json.dumps(body, ensure_ascii=False).encode()
                    update(request=dump(str(row["attempt"]) + "-request.json", blob))
                    if delivery is not None:
                        fingerprint = hashlib.sha256(
                            json.dumps(
                                body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest()
                        delivery.begin_attempt(lease, fingerprint, row["attempt"])
                        update(request_fingerprint=fingerprint)
                    count = counter(body["messages"], remaining())
                    update(input_tokens=count)
                    if type(count) is not int or count < 0:
                        raise ValueError("actual nonnegative tokenizer count required")
                    if count > request.model_profile.input_budget():
                        update(status="context_rejected_before_dispatch", http_status=400)
                        if delivery is not None:
                            delivery.rejected(lease, context=True)
                        return self.error(400, "context_length_exceeded;no generation dispatched")
                    headers = {"Content-Type": "application/json"}
                    if config.api_key:
                        headers["Authorization"] = "Bearer " + config.api_key
                    cap = remaining()
                    if delivery is not None:
                        delivery.dispatched(lease)
                    update(
                        dispatch_attempted=True,
                        status="dispatch_outcome_pending",
                        effective_timeout_s=cap,
                    )
                    save()
                    upstream = urllib.request.Request(url, data=blob, headers=headers)
                    with opener.open(upstream, timeout=cap) as response:
                        if delivery is None:
                            data = response.read(8 * 1024**2 + 1)
                        else:
                            chunks = []
                            size = 0
                            while size <= 8 * 1024**2:
                                left = remaining()
                                if response.fp is None:
                                    break
                                # This transport owns the urllib HTTPResponse.
                                # Bound each receive by the same absolute lease.
                                response.fp.raw._sock.settimeout(left)
                                piece = response.read1(min(65536, 8 * 1024**2 + 1 - size))
                                if not piece:
                                    break
                                chunks.append(piece)
                                size += len(piece)
                                if response.isclosed():
                                    break
                            data = b"".join(chunks)
                            if response.length not in (None, 0):
                                raise ValueError("incomplete HTTP Content-Length body")
                        status = response.status
                    if len(data) > 8 * 1024**2:
                        raise ValueError("response exceeded bounded complete payload")
                    parsed = json.loads(data)
                    update(response=dump(str(row["attempt"]) + "-response.json", data))
                    usage = parsed.get("usage")
                    if isinstance(usage, dict) and all(
                        type(usage.get(k)) is int and usage[k] >= 0
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    ):
                        update(usage=usage)
                    # The caller may bind a parser failure to this completed
                    # response immediately. Publish its receipt before returning
                    # bytes,so a valid observation is not lost to a file race.
                    update(status="response_received", http_status=status)
                    if delivery is not None and not delivery.delivered(lease, row.get("usage")):
                        update(status="late_response_accounted;no_model_action", http_status=status)
                        save()
                        return self.error(
                            400,
                            "logical query already ended;late usage recorded,no action delivered",
                        )
                    save()
                    self.reply(status, data)
                except urllib.error.HTTPError as error:
                    update(status="upstream_http_error;usage_unknown", http_status=error.code)
                    error.close()
                    if delivery is not None:
                        delivery.unknown(lease, "upstream HTTP error;dispatch was attempted")
                    self.error(error.code, "upstream HTTP error;see physical usage ledger")
                except (ValueError, TypeError, KeyError) as error:
                    update(
                        status="invalid_or_incomplete_response"
                        if row["dispatch_attempted"]
                        else "invalid_request",
                        error_type=type(error).__name__,
                        http_status=400,
                    )
                    if delivery is not None:
                        if row["dispatch_attempted"]:
                            delivery.unknown(lease, "invalid or incomplete upstream response")
                        else:
                            delivery.rejected(lease)
                    self.error(400, row["status"])
                except Exception as error:
                    update(
                        status="dispatch_outcome_unknown"
                        if row["dispatch_attempted"]
                        else "not_dispatched",
                        error_type=type(error).__name__,
                        http_status=504,
                    )
                    if delivery is not None:
                        from infinicoder.models.native_delivery_policy import (
                            NativeDeliveryExhausted,
                        )

                        if isinstance(error, NativeDeliveryExhausted):
                            pass
                        elif row["dispatch_attempted"]:
                            delivery.unknown(lease, "upstream transport failed after dispatch")
                        else:
                            delivery.not_dispatched(lease, type(error).__name__)
                    self.error(
                        504, "transport deadline or connection failure;unknown usage is not zero"
                    )
                finally:
                    update(wall_s=time.monotonic() - tick)
                    save()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        save()
        try:
            yield ModelBinding(
                config.model,
                dict(
                    api_base="http://127.0.0.1:" + str(server.server_port) + "/v1",
                    api_key=token,
                    temperature=config.temperature,
                ),
                str(ledger),
            )
        finally:
            closing = True
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            save()
            self._limits_binding = None
            self._physical_budget = None
            self._delivery = None
            self._request_observations = None
