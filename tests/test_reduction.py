"""Semantic admission controls;actual CUDA math is qualified separately."""

import ast
import unittest
from infinicoder.domains.operator.reduction import analyze, generate


def reference(operation="sum", axis="self.dim", keepdim="True", init="1"):
    return (
        "import torch\nimport torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, dim: int):\n"
        "        super().__init__()\n"
        "        self.dim = dim\n"
        "    def forward(self, x):\n"
        f"        return torch.{operation}(x, dim={axis}, keepdim={keepdim})\n"
        f"def get_init_inputs():\n    return [{init}]\n"
    )


class ReductionAdmissionTests(unittest.TestCase):
    def test_axis_config_negative_and_keepdim_are_preserved(self):
        for axis in ("self.dim", "-1", "0", "1"):
            for keep in ("True", "False"):
                descriptor = analyze(reference(axis=axis, keepdim=keep, init="-1"))
                self.assertEqual(descriptor.dimension_expression, axis)
                self.assertEqual(descriptor.keepdim, keep == "True")
                self.assertEqual(descriptor.constructor_values, (-1,))
                for schedule in ("scalar", "coalesced"):
                    candidate = generate(reference(axis=axis, keepdim=keep, init="-1"), schedule)
                    ast.parse(candidate.source)
                    self.assertEqual(candidate.schedule, schedule)

    def test_bound_tensor_method_and_module_alias(self):
        source = reference().replace("torch.sum(x, dim=self.dim", "x.sum(dim=self.dim")
        self.assertTrue(analyze(source).keepdim)
        source = (
            reference()
            .replace("import torch\n", "import torch as t\n")
            .replace("torch.sum", "t.sum")
        )
        self.assertEqual(analyze(source).dimension_expression, "self.dim")

    def test_distinct_semantics_and_unknown_bindings_are_not_lowered(self):
        sources = [
            reference("mean"),
            reference("amax"),
            reference(init="(0, 2)"),
            reference(axis="(0, 2)"),
            reference().replace("keepdim=True", "keepdim=self.dim"),
            reference().replace("keepdim=True", "keepdim=True, dtype=torch.float64"),
            reference().replace(
                "get_init_inputs():\n    return [1]",
                "get_init_inputs():\n    return [unknown_axis()]",
            ),
        ]
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaises((ValueError, TypeError)):
                    generate(source)

    def test_semantic_name_rebinding_and_extra_state_rejected(self):
        for source in [
            reference() + "torch = object()\n",
            reference().replace("self.dim = dim", "self.dim = dim\n        self.extra = dim"),
            reference().replace("def forward(self, x):", "def forward(self, torch):"),
        ]:
            with self.assertRaises(ValueError):
                analyze(source)

    def test_emitted_compute_has_no_torch_sum_fallback(self):
        for schedule in ("scalar", "coalesced"):
            candidate = generate(reference(), schedule)
            calls = [
                ast.unparse(n.func)
                for n in ast.walk(ast.parse(candidate.source))
                if isinstance(n, ast.Call)
            ]
            self.assertNotIn("torch.sum", calls)
            self.assertIn("tl.sum", calls)
            self.assertIn("torch.empty", calls)


if __name__ == "__main__":
    unittest.main()
