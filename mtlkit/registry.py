"""mtlkit/registry.py — one generic registry shared by every registry-backed seam.

Eng Review decision 3B (2026-09-06): tasks.py, pooling.py, combine.py, and
compose.py each need a name -> component lookup with a clear error on a typo'd
key. Rather than four independent dict-plus-lookup implementations, they all
share this one ``Registry[T]``.

    TASK_REGISTRY: Registry[TaskSpec] = Registry("task")
    TASK_REGISTRY.register("ks", TaskSpec(...))
    TASK_REGISTRY.get("ks")       # -> TaskSpec(...)
    TASK_REGISTRY.get("kx")       # -> KeyError: Unknown task 'kx'. Valid options: ...

Registration is permanent for the process lifetime (no unregister) and
duplicate keys are rejected — a silent overwrite would hide a copy-paste bug
in whichever module registers second.
"""

from typing import Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> component lookup with a descriptive error on a missing key."""

    def __init__(self, kind: str):
        """``kind`` names what this registry holds (e.g. "task", "pooling
        strategy") — it appears in every error message this registry raises."""
        self._kind = kind
        self._items: Dict[str, T] = {}

    def register(self, key: str, item: T) -> T:
        if key in self._items:
            raise KeyError(
                f"{self._kind} '{key}' is already registered "
                f"(existing keys: {', '.join(self.list())})"
            )
        self._items[key] = item
        return item

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            valid = ", ".join(self.list()) or "(none registered)"
            raise KeyError(
                f"Unknown {self._kind} '{key}'. Valid options: {valid}"
            ) from None

    def try_get(self, key: str) -> Optional[T]:
        """Like ``get``, but returns ``None`` on a missing key instead of
        raising — for callers that need a null-check, not an error (e.g.
        parsing a token that might legitimately not be a registered key)."""
        return self._items.get(key)

    def list(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.list())
