"""Helpers for `__torch_function__` dispatch enumeration tests."""
from __future__ import annotations

from typing import Iterable

import torch


# Ops we explicitly skip in dispatch fuzzing because they have side effects,
# require special-purpose inputs, or are clearly out of scope.
DISPATCH_SKIP = {
    "save", "load", "compile", "no_grad", "enable_grad", "set_grad_enabled",
    "manual_seed", "seed", "initial_seed", "get_rng_state", "set_rng_state",
    "use_deterministic_algorithms", "are_deterministic_algorithms_enabled",
    "set_default_dtype", "set_default_device", "set_default_tensor_type",
    "set_printoptions", "set_num_threads", "set_num_interop_threads",
    "get_default_dtype", "get_num_threads", "get_num_interop_threads",
    "autocast", "is_warn_always_enabled", "set_warn_always",
    "load_inline", "fork", "wait", "jit", "fx", "onnx", "library", "Tag",
    "is_grad_enabled", "is_inference_mode_enabled",
    "set_flush_denormal", "get_float32_matmul_precision",
    "set_float32_matmul_precision",
    "Generator", "Tensor", "Size", "Storage", "Stream", "device",
    "dtype", "layout", "memory_format",
    "ScriptFunction", "ScriptModule", "compile_for_static",
    "_C", "ops", "classes", "version",
}


def torch_unary_ops(allow_skip: Iterable[str] = ()) -> list[tuple[str, callable]]:
    """Enumerate likely-unary torch namespace ops for dispatch fuzzing."""
    skip = set(DISPATCH_SKIP) | set(allow_skip)
    out = []
    for name in dir(torch):
        if name.startswith("_") or name in skip:
            continue
        obj = getattr(torch, name)
        if not callable(obj):
            continue
        out.append((name, obj))
    return out
