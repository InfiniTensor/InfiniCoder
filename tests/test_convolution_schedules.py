"""Owned host-launch controls,no reference/GPU execution or performance claim."""

import ast
from contextlib import nullcontext
from dataclasses import replace
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import test_native_implementation_provider as fixture

from infinicoder.domains.operator.convolution_provider import assess
from infinicoder.domains.operator.convolution_schedules import (
    SCHEDULES,
    _render,
    propose_convolution_schedules,
)
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.execution.workspace_tools import Workspace


class ScheduleControls(unittest.TestCase):
    def test_actual_host_launch_grids_and_tiles_match_for_both_kernel_branches(self):
        with tempfile.TemporaryDirectory() as folder:
            c = fixture.ProviderControls().setup_case(Path(folder))
            workspace = Workspace(c.workspace, ("solution.py",))
            baseline = assess(c.request.spec, workspace, c.contract)["candidate_source"]
            bundle = propose_convolution_schedules(c.request.spec, workspace, c.contract)
            self.assertTrue(bundle.current_for(c.request.spec, c.contract))
            self.assertEqual(bundle.variants[0].source, baseline)
            self.assertEqual(len({v.sha256 for v in bundle.variants}), 3)

            class Tensor:
                dtype = "owned-float32"
                device = "owned-cuda"
                is_cuda = True
                requires_grad = False

                def __init__(self, shape):
                    self.shape = shape
                    self.ndim = len(shape)

                def stride(self, index=None):
                    values = tuple(math.prod(self.shape[i + 1 :]) for i in range(len(self.shape)))
                    return values if index is None else values[index]

                def numel(self):
                    return math.prod(self.shape)

            for depthwise in (False, True):
                co = 6 if depthwise else 64
                groups = 3 if depthwise else 1
                cg = 3 // groups
                module = SimpleNamespace(
                    kernel_size=(3, 3),
                    in_channels=3,
                    out_channels=co,
                    groups=groups,
                    weight=Tensor((co, cg, 3, 3)),
                    bias=None,
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    output_padding=(0, 0),
                    transposed=False,
                )
                x = Tensor((2, 3, 16, 19))
                for variant in bundle.variants:
                    with self.subTest(depthwise=depthwise, schedule=variant.schedule.name):
                        calls = []

                        class Kernel:
                            def __init__(self, name):
                                self.name = name

                            def __getitem__(self, grid):
                                def launch(*args, **kwargs):
                                    calls.append((self.name, grid, kwargs))

                                return launch

                        torch = SimpleNamespace(
                            float32="owned-float32",
                            is_grad_enabled=lambda: False,
                            empty=lambda shape, **kwargs: Tensor(shape),
                            cuda=SimpleNamespace(
                                get_device_capability=lambda device: (8, 0),
                                device=lambda device: nullcontext(),
                            ),
                        )
                        forward = next(
                            n
                            for n in ast.parse(variant.source).body
                            if isinstance(n, ast.FunctionDef) and n.name == "_ic_forward"
                        )
                        namespace = dict(
                            torch=torch,
                            math=math,
                            triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
                            _ic_conv=Kernel("conv"),
                            _ic_depthwise=Kernel("depthwise"),
                        )
                        exec(
                            compile(
                                ast.Module(body=[forward], type_ignores=[]),
                                "owned_generated_host_wrapper",
                                "exec",
                            ),
                            namespace,
                        )
                        result = namespace["_ic_forward"](x, module)
                        self.assertEqual(result.shape, (2, co, 14, 17))
                        self.assertEqual(len(calls), 1)
                        name, grid, keywords = calls[0]
                        s = variant.schedule
                        if depthwise:
                            self.assertEqual(
                                (name, grid),
                                (
                                    "depthwise",
                                    (
                                        (result.numel() + s.depthwise_block - 1)
                                        // s.depthwise_block,
                                    ),
                                ),
                            )
                            self.assertEqual(keywords, dict(BLOCK=s.depthwise_block, num_warps=4))
                        else:
                            self.assertEqual(
                                (name, grid),
                                (
                                    "conv",
                                    ((2 * 14 * 17 + s.bm - 1) // s.bm, (co + s.bn - 1) // s.bn, 1),
                                ),
                            )
                            self.assertEqual(keywords, dict(BM=s.bm, BN=s.bn, BK=32, num_warps=4))
            self.assertEqual(c.target.read_bytes(), fixture.STARTER)
            self.assertFalse(bundle.current_for(TaskSpec(c.task + " changed"), c.contract))

    def test_reference_outside_existing_grammar_stays_not_applicable(self):
        with tempfile.TemporaryDirectory() as folder:
            source = fixture.REFERENCE.replace(
                "nn.Conv2d(ci, co, kernel, bias=False)", "nn.AvgPool2d(kernel)"
            )
            c = fixture.ProviderControls().setup_case(Path(folder), reference=source)
            self.assertIsNone(
                propose_convolution_schedules(
                    c.request.spec, Workspace(c.workspace, ("solution.py",)), c.contract
                )
            )
            self.assertEqual(c.target.read_bytes(), fixture.STARTER)

    def test_unknown_schedule_or_changed_launch_template_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            c = fixture.ProviderControls().setup_case(Path(folder))
            baseline = assess(c.request.spec, Workspace(c.workspace, ("solution.py",)), c.contract)[
                "candidate_source"
            ]
            with self.assertRaises(ValueError):
                _render(baseline, replace(SCHEDULES[1], name="unfrozen"))
            with self.assertRaises(ValueError):
                _render(baseline.replace("BM=16", "BM=17"), SCHEDULES[1])


if __name__ == "__main__":
    unittest.main()
