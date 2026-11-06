"""Test-only: pin synthetic parts in `vm_judges` for launcher subprocesses (never shipped, never read by production code).

A counted window refuses a part that matches no pin in code (`vm_judges.pinned_real_parts`, `PART_NOT_PINNED`). The
free e2e tests drive the real `launch_judges.sh` on parts the real builder makes in a temp dir, whose hashes change
with every build, so they cannot be pinned in code. A test puts this directory first on `PYTHONPATH` and names a JSON
file `{pin_name: [part_json_sha256, items_jsonl_sha256]}` in `FAKE_COLAB_TEST_PART_PINS_FILE`; every Python the
launcher starts then loads this `sitecustomize`, which adds those pins to `pinned_real_parts` when `vm_judges` is
imported. The fake VM gets the same pins from the real-mode shim (`fake_vm_real_mode_shim.install`), because its cell
executes `vm_judges` without importing it. Production has no such hook: the only pins are the ones in code.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
from typing import Any

_PINS_FILE = os.environ.get("FAKE_COLAB_TEST_PART_PINS_FILE", "").strip()


def apply_test_part_pins(module_globals: dict[str, Any], pins_file: str) -> None:
    """Wrap `pinned_real_parts` in `module_globals` so it also returns the pins in `pins_file`."""
    with open(pins_file, encoding="utf-8") as fh:
        extra = {str(k): (str(v[0]), str(v[1])) for k, v in json.load(fh).items()}
    orig = module_globals["pinned_real_parts"]

    def pinned_real_parts() -> dict[str, tuple[str, str]]:
        pins = dict(orig())
        pins.update(extra)
        return pins

    module_globals["pinned_real_parts"] = pinned_real_parts
    # The test pins count as counted parts. The burned pilot can never be made acceptable for counted this way:
    # `require_pinned_part` refuses it by its hashes before it looks at pin names.
    orig_counted = module_globals["counted_part_pin_names"]
    module_globals["counted_part_pin_names"] = lambda: frozenset(orig_counted()) | frozenset(extra)


class _PinFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name != "vm_judges":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module: Any) -> None:
            orig_exec(module)
            apply_test_part_pins(module.__dict__, _PINS_FILE)

        loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


if _PINS_FILE:
    sys.meta_path.insert(0, _PinFinder())
