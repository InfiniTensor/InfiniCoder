"""Template emitted into a caller-authorized package;no Triton import on fallback.

Numerical reassociation is an explicit caller choice,not a guard-proven equality.
The backend's compile/launch/runtime errors are deliberately not swallowed.
"""

TEMPLATE = '''"""Explicit approximate FP32 reduction;other calls retain the original operation."""
import torch

NUMERICAL_CONTRACT = __NUMERICAL_CONTRACT__
_FAST = None


def fallback_reason(x, original_sum):
    # Check special execution contexts before touching tensor metadata.
    if type(x) is not torch.Tensor:
        return 'nonordinary_tensor'
    canonical = getattr(getattr(torch._C, '_VariableFunctions', None), 'sum', None)
    if canonical is None or original_sum is not canonical:
        return 'noncanonical_sum'
    try:
        from torch.overrides import _get_current_function_mode_stack
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
        if _get_current_function_mode_stack() or _get_current_dispatch_mode_stack():
            return 'special_dispatch'
        from torch.autograd import forward_ad
        if getattr(forward_ad, '_current_level', None) != -1:
            return 'forward_AD_or_unknown_context'
        if (not torch.is_inference_mode_enabled() or torch.is_grad_enabled()
                or torch._C._are_functorch_transforms_active()):
            return 'automatic_differentiation_context'
        if (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting()
                or torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu')
                or torch.are_deterministic_algorithms_enabled()):
            return 'special_execution_context'
    except (AttributeError, ImportError, TypeError):
        return 'unsupported_context_API'
    if (x.layout != torch.strided or x.dtype != torch.float32 or x.device.type != 'cuda'
            or x.requires_grad or x.is_nested or x.is_conj() or x.is_neg()):
        return 'tensor_domain'
    if any(name is not None for name in x.names):
        return 'named_tensor'
    axis = __AXIS__
    if x.ndim not in (2, 3) or not -x.ndim <= axis < x.ndim:
        return 'rank_or_axis'
    axis %= x.ndim
    if (any(type(n) is not int or n <= 0 for n in x.shape)
            or x.shape[axis] > 8192 or x.numel() > 67108864
            or any(s < 0 for s in x.stride())):
        return 'resource_domain'
    try:
        if x.device.index != torch.cuda.current_device() or torch.cuda.is_current_stream_capturing():
            return 'device_or_graph_context'
    except (AttributeError, TypeError):
        return 'unsupported_device_API'
    return None


def _backend():
    global _FAST
    if _FAST is None:
        try:
            from .__BACKEND_MODULE__ import ModelNew
        except ImportError:
            return None
        _FAST = ModelNew().forward
    return _FAST


def run(x, original_sum):
    if fallback_reason(x, original_sum) is not None:
        return __ORIGINAL_CALL__
    fast = _backend()
    if fast is None:
        return __ORIGINAL_CALL__
    # Once selected,any compile,launch or asynchronous failure belongs to the
    # candidate. Re-running the original op would hide failures and duplicate work.
    return fast(x)
'''
