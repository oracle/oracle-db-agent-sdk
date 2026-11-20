##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Prepare workflow definitions for database execution."""

import json
import math

from pyagentspec import AgentSpecSerializer
from pyagentspec.flows.node import Node

from ._pattern_compiler import build_standalone_node_flow
from ._wrappers import (
    Executable,
    _unwrap_executable,
)
from .exceptions import AgentSpecSerializationError


def _reject_nonfinite_constant(value: str) -> None:
    """Reject the nonstandard NaN/Infinity tokens accepted by Python's decoder."""
    raise ValueError("AgentSpec JSON contains a non-finite number.")


def _finite_float(value: str) -> float:
    """Reject JSON numbers whose conversion would overflow to infinity."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("AgentSpec JSON contains a non-finite number.")
    return result


def _relocate_parallel_llm_configs(
    value: object, enclosing_flow: dict[str, object] | None = None
) -> bool:
    """Move parallel-node LLM definitions into the enclosing database Flow scope."""
    changed = False
    if isinstance(value, list):
        for child in value:
            changed = _relocate_parallel_llm_configs(child, enclosing_flow) or changed
    elif isinstance(value, dict):
        component_type = value.get("component_type")
        if component_type == "Flow":
            enclosing_flow = value
        elif component_type == "DbmsVectorChainLlmConfig":
            return False

        if component_type == "ParallelFlowNode":
            definitions = value.get("$referenced_components")
            if isinstance(definitions, dict):
                for reference_id, definition in list(definitions.items()):
                    if not isinstance(definition, dict) or (
                        definition.get("component_type") != "DbmsVectorChainLlmConfig"
                    ):
                        continue
                    if enclosing_flow is None:
                        raise ValueError("Shared model configuration has no enclosing Flow.")
                    destination = enclosing_flow.setdefault("$referenced_components", {})
                    if not isinstance(destination, dict) or reference_id in destination:
                        raise ValueError("Shared model configuration reference conflicts.")
                    destination[reference_id] = definition
                    del definitions[reference_id]
                    changed = True
                if changed and not definitions:
                    del value["$referenced_components"]

        for key, child in list(value.items()):
            if key not in {"inputs", "outputs", "metadata", "configuration"}:
                changed = _relocate_parallel_llm_configs(child, enclosing_flow) or changed
    return changed


def _serialize_workflow(workflow: Executable) -> str:
    """Prepare an SDK workflow for execution without exposing invalid payload details."""
    executable = _unwrap_executable(workflow)
    if isinstance(executable, Node):
        executable = build_standalone_node_flow(executable)

    try:
        agent_spec_json = AgentSpecSerializer().to_json(executable)
        if not isinstance(agent_spec_json, str):
            raise TypeError("AgentSpecSerializer returned an unexpected value")
        # Validate numbers before adapting component definitions for the database.
        document = json.loads(
            agent_spec_json,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_finite_float,
        )
        changed = _relocate_parallel_llm_configs(document)

        def omit_empty_transforms(value: object) -> None:
            nonlocal changed
            if isinstance(value, dict):
                if value.get("component_type") == "Agent" and value.get("transforms") == []:
                    del value["transforms"]
                    changed = True
                # Contracts, metadata and executor configuration contain application
                # data, not graph components. Never rewrite their defaults or schemas.
                for key, child in value.items():
                    if key not in {"inputs", "outputs", "metadata", "configuration"}:
                        omit_empty_transforms(child)
            elif isinstance(value, list):
                for child in value:
                    omit_empty_transforms(child)

        omit_empty_transforms(document)
        if changed:
            return json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return agent_spec_json
    except Exception:
        # Serializer diagnostics may contain the workflow payload.
        raise AgentSpecSerializationError(
            "Workflow could not be serialized as AgentSpec JSON."
        ) from None
