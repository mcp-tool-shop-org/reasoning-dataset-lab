"""A small, strict JSON-Schema checker for the public files of a sealed Pilot B part.

The schemas ship in ``schemas/`` beside this module. They are closed: ``additionalProperties`` is false wherever an
object has a fixed shape, and arrays whose length is fixed say so. ``verify_sealed_part`` validates every public file
against them, so a field, an entry or a key that the writer does not produce fails verification.

Only the keywords the schemas use are supported: ``$ref`` (to ``#/$defs/...``), ``type``, ``properties``,
``required``, ``additionalProperties``, ``items``, ``minItems``, ``maxItems``, ``enum``, ``const``, ``pattern``,
``minimum`` and ``anyOf``. An unknown keyword is an error, so a schema cannot silently stop checking something.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
_KNOWN = {
    "$ref",
    "$defs",
    "$schema",
    "title",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "pattern",
    "minimum",
    "anyOf",
}
_TYPES: dict[str, Any] = {
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, Sequence) and not isinstance(v, (str, bytes)),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


@cache
def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def validate(instance: Any, schema_name: str) -> list[str]:
    """Errors (as ``path: message``) of ``instance`` against the named shipped schema; empty when it conforms."""

    root = load_schema(schema_name)
    return _check(instance, root, root, "$")


def _check(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> list[str]:
    unknown = set(schema) - _KNOWN
    if unknown:
        return [f"{path}: schema keyword not supported: {sorted(unknown)[0]}"]
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if not ref.startswith("#/$defs/"):
            return [f"{path}: unsupported reference {ref}"]
        return _check(value, root["$defs"][ref[len("#/$defs/") :]], root, path)
    if "anyOf" in schema:
        attempts = [_check(value, sub, root, path) for sub in schema["anyOf"]]
        return [] if any(not e for e in attempts) else [f"{path}: matches none of the allowed shapes"]
    errors: list[str] = []
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](value) for t in allowed):
            return [f"{path}: expected {'|'.join(allowed)}"]
    if "const" in schema and (value != schema["const"] or type(value) is not type(schema["const"])):
        errors.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not one of {schema['enum']}")
    if "pattern" in schema and isinstance(value, str) and re.search(schema["pattern"], value) is None:
        errors.append(f"{path}: does not match {schema['pattern']}")
    if "minimum" in schema and _TYPES["number"](value) and value < schema["minimum"]:
        errors.append(f"{path}: below {schema['minimum']}")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing {name}")
        extra = schema.get("additionalProperties", True)
        for name, member in value.items():
            if name in properties:
                errors.extend(_check(member, properties[name], root, f"{path}.{name}"))
            elif extra is False:
                errors.append(f"{path}: unexpected field {name}")
            elif isinstance(extra, Mapping):
                errors.extend(_check(member, extra, root, f"{path}.{name}"))
    elif _TYPES["array"](value) and not isinstance(value, str):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} entries")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} entries")
        if "items" in schema:
            for index, member in enumerate(value):
                errors.extend(_check(member, schema["items"], root, f"{path}[{index}]"))
    return errors
