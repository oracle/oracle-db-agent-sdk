##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Build workflows that repeat a step within an iteration limit."""

from __future__ import annotations

from copy import deepcopy

from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import BranchingNode, EndNode, StartNode

from ._contracts import preserve_contract_schemas, properties_have_same_type
from ._flow_status import fresh_copy, guarded_flow, public_outputs
from ._json_types import JSONValue
from ._pattern_compiler import _make_routing_node
from ._serialization import _serialize_workflow
from ._wrappers import RuntimeComponent, _wrap_executable
from .exceptions import IncompatibleLoopContractError, PatternCompilationError

_STATE_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}


def _loop_state_contract(
    component: RuntimeComponent, *, state_key: str
) -> tuple[dict[str, JSONValue], dict[str, tuple[str, ...]]]:
    """Validate feedback and describe the public state object and runtime types."""
    inputs = {prop.title: prop for prop in component.inputs or ()}
    outputs = {prop.title: prop for prop in public_outputs(component)}
    if (
        not inputs
        or len(inputs) != len(component.inputs or ())
        or len(outputs) != len(public_outputs(component))
        or inputs.keys() != outputs.keys()
    ):
        raise IncompatibleLoopContractError(
            "Loop body inputs and outputs must have the same nonempty, unique names."
        )
    if any(not properties_have_same_type(inputs[key], outputs[key]) for key in inputs):
        raise IncompatibleLoopContractError("Loop body feedback property types must match.")
    if state_key not in inputs or inputs[state_key].type != "string":
        raise IncompatibleLoopContractError("Loop state_key must name a string body state field.")

    state_types: dict[str, tuple[str, ...]] = {}
    fields: dict[str, JSONValue] = {}
    for title, prop in inputs.items():
        types = [prop.type] if isinstance(prop.type, str) else prop.type
        if not types or not set(types) <= _STATE_TYPES:
            raise IncompatibleLoopContractError("Loop state fields need explicit JSON types.")
        state_types[title] = tuple(types)
        fields[title] = deepcopy(prop.json_schema)
    return (
        {
            "type": "object",
            "properties": fields,
            "required": list(inputs),
        },
        state_types,
    )


def build_loop_flow(
    component: RuntimeComponent,
    *,
    name: str,
    state_key: str,
    continue_value: str,
    max_iterations: int,
) -> Flow:
    """Validate feedback contracts and define a bounded repeating workflow."""
    try:
        component = fresh_copy(component)
        _loop_state_contract(component, state_key=state_key)
        _serialize_workflow(_wrap_executable(deepcopy(component)))
        with preserve_contract_schemas(component):
            fields = deepcopy(list(component.inputs or ()))
            selector = next(field for field in fields if field.title == state_key)
            with preserve_contract_schemas(fields):
                start = StartNode(name="loop-start", inputs=fields)
            gates = [
                BranchingNode(
                    name=f"loop-condition-{index}",
                    inputs=[selector],
                    mapping={continue_value: "loop_continue"},
                )
                for index in range(max_iterations + 1)
            ]
            bodies = []
            for index in range(max_iterations):
                # Separate call sites preserve iteration identity; sharing the
                # definition lets the serializer emit it once with references.
                with preserve_contract_schemas(component):
                    bodies.append(_make_routing_node(component, name=f"loop-body-{index}"))
            output_fields = deepcopy(public_outputs(component))
            with preserve_contract_schemas(output_fields):
                end = EndNode(name="loop-end", outputs=output_fields)
            nodes = [start, *gates, *bodies, end]
            control = [ControlFlowEdge(name="loop-enter", from_node=start, to_node=gates[0])]
            data: list[DataFlowEdge] = []
            # Each source supplies the state for one gate. Gates inspect only
            # the selector; bodies and the successful exit receive all fields.
            for index, source in enumerate([start, *bodies]):
                connections = [(gates[index], [selector]), (end, fields)]
                control.append(
                    ControlFlowEdge(
                        name=f"loop-exit-{index}",
                        from_node=gates[index],
                        from_branch="default",
                        to_node=end,
                    )
                )
                if index < max_iterations:
                    body = bodies[index]
                    connections.append((body, fields))
                    control.append(
                        ControlFlowEdge(
                            name=f"loop-continue-{index}",
                            from_node=gates[index],
                            from_branch="loop_continue",
                            to_node=body,
                        )
                    )
                    for branch in body.branches or ["next"]:
                        control.append(
                            ControlFlowEdge(
                                name=f"loop-forward-{index}-{branch}",
                                from_node=body,
                                from_branch=branch,
                                to_node=gates[index + 1],
                            )
                        )
                for destination, properties in connections:
                    for field in properties:
                        data.append(
                            DataFlowEdge(
                                name=f"loop-state-{len(data)}",
                                source_node=source,
                                source_output=field.title,
                                destination_node=destination,
                                destination_input=field.title,
                            )
                        )
            return guarded_flow(
                name=name,
                start_node=start,
                nodes=nodes,
                outputs=public_outputs(component),
                control_flow_connections=control,
                data_flow_connections=data,
                exhaustion=[(gates[-1], "loop_continue")],
            )
    except PatternCompilationError:
        raise
    except Exception:
        raise PatternCompilationError(
            "Loop could not be compiled into an AgentSpec Flow."
        ) from None
