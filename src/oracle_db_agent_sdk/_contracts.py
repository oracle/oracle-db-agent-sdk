##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Validate and preserve input and output contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, TypeAlias

from pyagentspec.component import Component
from pyagentspec.property import Property
from pyagentspec.property import properties_have_same_type as _properties_have_same_type

from ._json_types import JSONValue
from .exceptions import BuildingBlockError

PropertySchema: TypeAlias = str | Mapping[str, JSONValue]
Contract: TypeAlias = Mapping[str, PropertySchema]


def properties_have_same_type(first: Property, second: Property) -> bool:
    """Compare disposable copies because PyAgentSpec may normalize schemas in place."""
    return _properties_have_same_type(first.model_copy(deep=True), second.model_copy(deep=True))


@contextmanager
def preserve_contract_schemas(*values: object) -> Iterator[None]:
    """Preserve declared schemas while validating workflow components."""
    pending = list(values)
    seen: set[int] = set()
    schemas: list[tuple[Property, dict[str, Any]]] = []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Property):
            schemas.append((value, deepcopy(value.json_schema)))
        elif isinstance(value, Component):
            pending.extend(getattr(value, field) for field in type(value).model_fields)
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    try:
        yield
    finally:
        for property_, schema in schemas:
            property_.json_schema = schema


def contract_properties(contract: Contract) -> list[Property]:
    """Convert a named contract to PyAgentSpec properties without mutating it.

    Accept JSON type strings or schema mappings. The contract key supplies the
    property title even if the schema defines its own title. Invalid definitions
    raise BuildingBlockError without exposing raw schema-validation details.
    """
    try:
        properties: list[Property] = []
        for property_name, property_schema in contract.items():
            if isinstance(property_schema, str):
                normalized_schema: dict[str, JSONValue] = {
                    "title": property_name,
                    "type": property_schema,
                }
            elif isinstance(property_schema, Mapping):
                normalized_schema = deepcopy(dict(property_schema))
                normalized_schema["title"] = property_name
            else:
                raise TypeError(
                    f"Contract property {property_name!r} must use a type string or JSON Schema."
                )
            # Default-value validation can normalize a schema in place too.
            property_ = Property(json_schema=deepcopy(normalized_schema))
            property_.json_schema = normalized_schema
            properties.append(property_)
        return properties
    except BuildingBlockError:
        raise
    except Exception:
        # Validation errors can include schema defaults and other supplied values.
        raise BuildingBlockError("The input or output contract is invalid.") from None
