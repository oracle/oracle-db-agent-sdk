##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Preserve workflow results and propagate execution failures."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, TypeVar
from uuid import uuid4

from pyagentspec.component import Component
from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.node import Node
from pyagentspec.flows.nodes import BranchingNode, EndNode, FlowNode, ParallelFlowNode, StartNode
from pyagentspec.property import Property

from ._contracts import preserve_contract_schemas
from ._wrappers import RuntimeComponent

_STATUS = "oracle_db_agent_sdk_loop_status"
_OUTPUTS = "oracle_db_agent_sdk_public_outputs"
_FAILURES = "oracle_db_agent_sdk_failure_statuses"
_C = TypeVar("_C", bound=Component)


def fresh_copy(component: _C) -> _C:
    """Copy an occurrence with fresh IDs, retaining references within that occurrence.

    Memoize by Python object identity so node lists and edges that reference the
    same component still reference one copy. New IDs isolate a loop's reusable
    body definition from definitions in other compiled loops.
    """
    memo: dict[int, Any] = {}

    def copy(value: Any) -> Any:
        if isinstance(value, Component):
            if id(value) not in memo:
                updates = {key: copy(getattr(value, key)) for key in type(value).model_fields}
                updates["id"] = str(uuid4())
                memo[id(value)] = value.model_copy(update=updates)
            return memo[id(value)]
        if isinstance(value, dict):
            return {key: copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(copy(item) for item in value)
        return deepcopy(value)

    result: _C = copy(component)
    return result


def status_field(component: RuntimeComponent) -> str | None:
    """Return only an SDK-generated status field, never infer one from its prefix."""
    if isinstance(component, FlowNode):
        component = component.subflow
    value = (component.metadata or {}).get(_STATUS)
    return value if isinstance(value, str) else None


def public_outputs(component: RuntimeComponent) -> list[Property]:
    """Recover original contracts, excluding private transport fields and defaults."""
    if isinstance(component, FlowNode):
        component = component.subflow
    schemas = (component.metadata or {}).get(_OUTPUTS)
    if status_field(component) and isinstance(schemas, list):
        return [_property(schema) for schema in schemas]
    if isinstance(component, ParallelFlowNode):
        return [prop for flow in component.subflows for prop in public_outputs(flow)]
    return list(component.outputs or ())


def failure_statuses(component: RuntimeComponent) -> set[str]:
    """Recover the failure kinds a guarded graph can return to its parent."""
    if isinstance(component, FlowNode):
        component = component.subflow
    if isinstance(component, ParallelFlowNode):
        return {value for flow in component.subflows for value in failure_statuses(flow)}
    if status_field(component) is None:
        return set()
    return set((component.metadata or {}).get(_FAILURES, ["exhausted"]))


def _property(schema: dict[str, Any]) -> Property:
    """Retain schemas that dependency default validation may normalize in place."""
    result = Property(json_schema=deepcopy(schema))
    result.json_schema = deepcopy(schema)
    return result


def _typed_placeholder(schema: dict[str, Any]) -> Any:
    """Create a type-compatible transport value; it is discarded on exhaustion."""
    for key in ("default", "const"):
        if key in schema:
            return deepcopy(schema[key])
    if schema.get("enum"):
        return deepcopy(schema["enum"][0])
    if schema.get("anyOf"):
        return _typed_placeholder(schema["anyOf"][0])
    kind = schema.get("type", "null")
    if isinstance(kind, list):
        kind = "null" if "null" in kind else kind[0]
    if kind == "object":
        return {
            key: _typed_placeholder(value) for key, value in schema.get("properties", {}).items()
        }
    return {"string": "", "number": 0, "integer": 0, "boolean": False, "array": [], "null": None}[
        kind
    ]


def _transport_output(prop: Property) -> Property:
    """Provide fallback values for failure responses without changing successful results."""
    schema = deepcopy(prop.json_schema)
    schema.setdefault("default", _typed_placeholder(schema))
    return _property(schema)


def guarded_flow(
    *,
    name: str,
    start_node: StartNode,
    nodes: list[Node],
    control_flow_connections: list[ControlFlowEdge],
    data_flow_connections: list[DataFlowEdge],
    outputs: Sequence[Property],
    exhaustion: Sequence[tuple[Node, str]] = (),
    failure_exits: Sequence[tuple[Node, str, str]] = (),
) -> Flow:
    """Propagate child workflow failures and enforce configured execution limits."""
    sources: list[tuple[Node, list[str]]] = []
    for node in nodes:
        fields = (
            [status_field(flow) for flow in node.subflows]
            if isinstance(node, ParallelFlowNode)
            else [status_field(node)]
        )
        statuses = [field for field in fields if field is not None]
        if statuses:
            sources.append((node, statuses))
    if not sources and not exhaustion and not failure_exits:
        return Flow(
            name=name,
            start_node=start_node,
            nodes=nodes,
            control_flow_connections=control_flow_connections,
            data_flow_connections=data_flow_connections,
        )

    occupied = {p.title for node in nodes for p in [*(node.inputs or ()), *(node.outputs or ())]}
    field = "__sdk_loop_status_" + uuid4().hex
    while field in occupied:
        field += "_"
    complete = Property(title=field, json_schema={"type": "string", "default": "complete"})
    failed = Property(title=field, json_schema={"type": "string", "default": "exhausted"})
    for node in nodes:
        if isinstance(node, EndNode):
            node.outputs = [*list(node.outputs or ()), complete]
            node.inputs = list(node.outputs)
    errors = {"exhausted"} if exhaustion else set()
    errors.update(value for _, _, value in failure_exits)
    errors.update(value for source, _ in sources for value in failure_statuses(source))
    error_ends = {}
    for value in sorted(errors):
        error_prop = failed.model_copy(
            update={
                "default": value,
                "json_schema": {
                    **failed.json_schema,
                    "default": value,
                },
            }
        )
        error_end = EndNode(
            name="loop-exhausted" if value == "exhausted" else value,
            branch_name="sdk_loop_" + value,
            outputs=[error_prop],
        )
        nodes.append(error_end)
        error_ends[value] = error_end

    def connect(source: Node, destination: Node, branch: str | None) -> None:
        control_flow_connections.append(
            ControlFlowEdge(
                name=f"loop-status-control-{len(control_flow_connections)}",
                from_node=source,
                from_branch=branch,
                to_node=destination,
            )
        )

    for source, child_fields in sources:
        # Rewire control only: existing data edges still carry successful child
        # outputs, but the destination cannot run until all status checks pass.
        outgoing = [edge for edge in control_flow_connections if edge.from_node is source]
        destinations = {edge.to_node.id for edge in outgoing}
        if len(destinations) != 1:
            raise ValueError("SDK status sources must have one continuation destination")
        destination = outgoing[0].to_node
        outgoing_ids = {id(edge) for edge in outgoing}
        control_flow_connections[:] = [
            edge for edge in control_flow_connections if id(edge) not in outgoing_ids
        ]
        checks = []
        for child_field in child_fields:
            prop = next(p for p in source.outputs or () if p.title == child_field)
            check = BranchingNode(
                name=f"loop-status-check-{len(nodes)}",
                inputs=[prop],
                mapping={value: value for value in sorted(errors)},
            )
            nodes.append(check)
            checks.append(check)
            data_flow_connections.append(
                DataFlowEdge(
                    name=f"loop-status-data-{len(data_flow_connections)}",
                    source_node=source,
                    source_output=child_field,
                    destination_node=check,
                    destination_input=child_field,
                )
            )
            for value, error_end in error_ends.items():
                connect(check, error_end, value)
        for edge in outgoing:
            connect(source, checks[0], edge.from_branch or Node.DEFAULT_NEXT_BRANCH)
        for index, check in enumerate(checks):
            connect(check, checks[index + 1] if index + 1 < len(checks) else destination, "default")
    for source, branch in exhaustion:
        connect(source, error_ends["exhausted"], branch)
    for source, branch, value in failure_exits:
        connect(source, error_ends[value], branch)
    # An exhausted End has no application outputs. Defaults let the Flow declare
    # a common output shape; status checks and the executor discard these values.
    transport = [*[_transport_output(prop) for prop in outputs], complete]
    with preserve_contract_schemas(nodes, transport):
        return Flow(
            name=name,
            start_node=start_node,
            nodes=nodes,
            outputs=transport,
            control_flow_connections=control_flow_connections,
            data_flow_connections=data_flow_connections,
            metadata={
                _STATUS: field,
                _OUTPUTS: [deepcopy(prop.json_schema) for prop in outputs],
                _FAILURES: sorted(errors),
            },
        )
