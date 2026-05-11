from __future__ import annotations

from typing import Callable, Iterator, Tuple, Dict, Optional, Set
from collections import OrderedDict
from abc import abstractmethod

from brute.tensor import Tensor


class Module(object):
    """
    Base class for all Brute modules.

    Subclass and implement forward(). Register parameters via
    register_parameter() and child modules via add_module() or by assigning
    a Module instance as an attribute.
    """

    def __init__(self) -> None:
        self._params: Dict[str, Tensor] = {}
        self._modules: Dict[str, Module] = {}
        self._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._forward_hooks: Dict[int, Callable] = OrderedDict()
        self._training: bool = True

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor:
        # Pre-hooks run before forward; a non-None return replaces the input.
        for hook in self._forward_pre_hooks.values():
            result = hook(self, x)
            if result is not None:
                x = result

        output = self.forward(x)

        # Post-hooks run after forward; a non-None return replaces the output.
        for hook in self._forward_hooks.values():
            result = hook(self, x, output)
            if result is not None:
                output = result

        return output

    def register_forward_pre_hook(self, hook: Callable) -> None:
        """
        Register hook(module, input) -> input | None called before forward().

        If the hook returns a non-None value it replaces the input to forward().
        Deregister with deregister_forward_pre_hook(hook).
        """
        self._forward_pre_hooks[id(hook)] = hook
    
    def deregister_forward_pre_hook(self, hook: Callable) -> None:
        """Deregister existing pre-forward hook method `hook` from this module."""
        del self._forward_pre_hooks[id(hook)]
    
    def deregister_forward_pre_hooks(self) -> None:
        """Deregisters all existing pre-forward hooks from this module."""
        self._forward_pre_hooks.clear()

    def register_forward_hook(self, hook: Callable) -> None:
        """
        Register hook(module, input, output) -> output | None called after forward().

        If the hook returns a non-None value it replaces the output of __call__.
        Deregister with deregister_forward_hook(hook).
        """
        self._forward_hooks[id(hook)] = hook
    
    def deregister_forward_hook(self, hook: Callable) -> None:
        """Deregister existing forward hook method `hook` from this module."""
        del self._forward_hooks[id(hook)]
    
    def deregister_forward_hooks(self) -> None:
        """Deregisters all existing forward hooks from this module."""
        self._forward_hooks.clear()

    def register_parameter(self, name: str, data: Tensor) -> None:
        """Add a Tensor as a named parameter of this module."""
        self._params[name] = data

    def get_parameter(self, target: str) -> Tensor:
        """Return the parameter at dot-separated path target."""
        parts = target.split('.')
        mod = self.get_submodule('.'.join(parts[:-1])) if len(parts) > 1 else self
        param_name = parts[-1]
        if param_name not in mod._params:
            raise AttributeError(f"'{type(mod).__name__}' has no parameter '{param_name}'")
        return mod._params[param_name]

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Yield (name, parameter) pairs, recursing into submodules when recurse=True."""
        for name, param in self._params.items():
            yield (f"{prefix}.{name}" if prefix else name), param
        if recurse:
            for mod_name, mod in self._modules.items():
                subprefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from mod.named_parameters(subprefix, recurse=True)

    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        """Yield all parameters, recursing into submodules when recurse=True."""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def add_module(self, name: str, module: Module) -> None:
        """Register a child module under name."""
        self._modules[name] = module

    def get_submodule(self, target: str) -> Module:
        """Return the submodule at dot-separated path target."""
        if not target:
            return self
        mod = self
        for part in target.split('.'):
            if part not in mod._modules:
                raise AttributeError(f"'{type(mod).__name__}' has no submodule '{part}'")
            mod = mod._modules[part]
        return mod

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        """Yield (name, module) for immediate children only."""
        yield from self._modules.items()

    def children(self) -> Iterator[Module]:
        """Yield immediate child modules."""
        for _, mod in self.named_children():
            yield mod

    def named_modules(self, memo: Optional[Set[int]] = None, prefix: str = '') -> Iterator[Tuple[str, Module]]:
        """Yield (name, module) for self and all descendants, skipping already-visited modules."""
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            yield prefix, self
            for name, mod in self._modules.items():
                subprefix = f"{prefix}.{name}" if prefix else name
                yield from mod.named_modules(memo, subprefix)

    def modules(self) -> Iterator[Module]:
        """Yield self and all descendant modules."""
        for _, mod in self.named_modules():
            yield mod

    def apply(self, fn: Callable[[Module], None]) -> Module:
        """Apply fn depth-first to every submodule then self; returns self."""
        for mod in self.children():
            mod.apply(fn)
        fn(self)
        return self

    def train(self) -> Module:
        """Set this module and all descendants to training mode."""
        self._training = True
        for mod in self._modules.values():
            mod.train()
        return self

    def eval(self) -> Module:
        """Set this module and all descendants to evaluation mode."""
        self._training = False
        for mod in self._modules.values():
            mod.eval()
        return self

    def state_dict(self, prefix: str = '') -> Dict[str, Tensor]:
        """Return a flat dict mapping prefixed parameter names to tensors."""
        result: Dict[str, Tensor] = {}
        for name, param in self._params.items():
            result[f"{prefix}{name}"] = param
        for mod_name, mod in self._modules.items():
            result.update(mod.state_dict(prefix=f"{prefix}{mod_name}."))
        return result

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True) -> None:
        """
        Copy parameters from state_dict into this module.

        With strict=True (default) raises RuntimeError if keys don't match exactly.
        """
        if strict:
            current_keys = set(self.state_dict().keys())
            incoming_keys = set(state_dict.keys())
            missing = current_keys - incoming_keys
            unexpected = incoming_keys - current_keys
            if missing or unexpected:
                raise RuntimeError(
                    f"Missing keys: {sorted(missing)}. Unexpected keys: {sorted(unexpected)}."
                )
        self._load_from_state_dict(state_dict, prefix='')

    def _load_from_state_dict(self, state_dict: Dict[str, Tensor], prefix: str) -> None:
        for name in self._params:
            key = f"{prefix}{name}"
            if key in state_dict:
                self._params[name] = state_dict[key]
        for mod_name, mod in self._modules.items():
            mod._load_from_state_dict(state_dict, f"{prefix}{mod_name}.")

    def extra_repr(self) -> str:
        """Override to return a string of per-instance fields shown in __repr__."""
        return ''

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> object:
        modules = self.__dict__.get('_modules', {})
        if name in modules:
            return modules[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        extra = self.extra_repr()
        children = [
            f'{" " * 2}({name}): {repr(mod).replace("\n", "\n" + (" " * 2))}'
            for name, mod in self._modules.items()
        ]
        out = f'{type(self).__name__}'
        if extra:
            out += f'({'\n  ' if '\n' in extra else ''}{extra.replace("\n", "\n" + (" " * 2))}{'\n' if '\n' in extra else ''})'
        if children:
            out += f'(\n{"\n".join(children)}\n)'
        elif not extra:
            out += '()'
        return out