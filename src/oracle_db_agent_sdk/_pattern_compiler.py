##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Validate and compose workflow steps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from uuid import uuid4

from pyagentspec import Agent
from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.node import Node
from pyagentspec.flows.nodes import (
    AgentNode,
    BranchingNode,
    EndNode,
    FlowNode,
    ParallelFlowNode,
    StartNode,
)
from pyagentspec.property import Property

from ._contracts import preserve_contract_schemas, properties_have_same_type
from ._flow_status import guarded_flow, public_outputs
from ._wrappers import RuntimeComponent
from .exceptions import (
    IncompatibleParallelContractError,
    IncompatibleRoutingContractError,
    IncompatibleSequentialContractError,
    PatternCompilationError,
)


def _end_node(
    *, name: str, outputs: list[Property], branch_name: str = Node.DEFAULT_NEXT_BRANCH
) -> EndNode:
    """Preserve declared output schemas during validation."""
    with preserve_contract_schemas(outputs):
        return EndNode(name=name, outputs=outputs, branch_name=branch_name)


def build_standalone_node_flow(node: Node) -> Flow:
    """Copy a standalone node into a flow with matching start and end contracts."""
    try:
        node = deepcopy(node)
        with preserve_contract_schemas(node):
            start_node = StartNode(
                name="standalone-start",
                inputs=list(node.inputs or ()),
            )
            executable_node = node
            end_node = _end_node(
                name="standalone-end",
                outputs=list(node.outputs or ()),
            )
            nodes = [start_node, executable_node, end_node]

            return Flow(
                name=f"{node.name}-workflow",
                start_node=start_node,
                nodes=nodes,
                control_flow_connections=_control_edges(nodes),
                data_flow_connections=_data_edges(nodes),
            )
    except Exception:
        raise PatternCompilationError(
            f"Standalone node '{node.name}' could not be compiled into a workflow."
        ) from None


def _properties_by_title(
    properties: Iterable[Property],
    *,
    component_name: str,
    direction: str,
) -> dict[str, Property]:
    """Index sequential properties by title, rejecting ambiguous duplicate names."""
    properties_by_title: dict[str, Property] = {}
    duplicates: list[str] = []
    for property_ in properties:
        if property_.title in properties_by_title:
            duplicates.append(property_.title)
        properties_by_title[property_.title] = property_
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise IncompatibleSequentialContractError(
            f"Sequential step '{component_name}' has duplicate {direction} properties: "
            f"{duplicate_list}."
        )
    return properties_by_title


def _validate_sequential_contract(
    source: RuntimeComponent,
    destination: RuntimeComponent,
) -> None:
    """Require exact adjacent property names and compatible PyAgentSpec types."""
    outputs = _properties_by_title(
        public_outputs(source), component_name=source.name, direction="output"
    )
    inputs = _properties_by_title(
        destination.inputs or (), component_name=destination.name, direction="input"
    )

    output_names = set(outputs)
    input_names = set(inputs)
    if output_names != input_names:
        missing = sorted(input_names - output_names)
        additional = sorted(output_names - input_names)
        details: list[str] = []
        if missing:
            details.append(f"missing outputs {missing}")
        if additional:
            details.append(f"additional outputs {additional}")
        raise IncompatibleSequentialContractError(
            f"Sequential step '{source.name}' cannot feed '{destination.name}': "
            + "; ".join(details)
            + "."
        )

    incompatible = [
        title
        for title in sorted(output_names)
        if not properties_have_same_type(outputs[title], inputs[title])
    ]
    if incompatible:
        raise IncompatibleSequentialContractError(
            f"Sequential step '{source.name}' cannot feed '{destination.name}': "
            f"incompatible property types for {incompatible}."
        )


def _make_sequential_step_node(
    component: RuntimeComponent,
    *,
    position: int,
) -> Node:
    """Adapt a component to a uniquely named step without changing the original."""
    node_name = f"sequential-step-{position:03d}-{component.name}"
    if isinstance(component, Flow):
        return FlowNode(name=node_name, subflow=component)
    if isinstance(component, Agent):
        return AgentNode(name=node_name, agent=component)
    if isinstance(component, Node):
        return component.model_copy(update={"id": str(uuid4()), "name": node_name})
    raise PatternCompilationError(f"Sequential component '{component.name}' is not supported.")


def _control_edges(
    nodes: Sequence[Node],
) -> list[ControlFlowEdge]:
    """Connect every outgoing branch of each node to the next node in order."""
    edges: list[ControlFlowEdge] = []
    for position, (source, destination) in enumerate(zip(nodes, nodes[1:], strict=False)):
        branches = sorted(source.branches or [Node.DEFAULT_NEXT_BRANCH])
        for branch in branches:
            is_default = branch == Node.DEFAULT_NEXT_BRANCH
            branch_name = "default" if is_default else branch
            edges.append(
                ControlFlowEdge(
                    name=f"sequential-control-{position:03d}-{branch_name}",
                    from_node=source,
                    from_branch=None if is_default and len(branches) == 1 else branch,
                    to_node=destination,
                )
            )
    return edges


def _data_edges(
    nodes: Sequence[Node],
) -> list[DataFlowEdge]:
    """Connect adjacent, compatible contracts by matching property titles."""
    edges: list[DataFlowEdge] = []
    for position, (source, destination) in enumerate(zip(nodes, nodes[1:], strict=False)):
        output_names = sorted(property_.title for property_ in public_outputs(source))
        for title in output_names:
            edges.append(
                DataFlowEdge(
                    name=f"sequential-data-{position:03d}-{title}",
                    source_node=source,
                    source_output=title,
                    destination_node=destination,
                    destination_input=title,
                )
            )
    return edges


def build_sequential_flow(
    components: Sequence[RuntimeComponent],
    *,
    name: str = "Sequential",
) -> Flow:
    """Validate and connect a sequence of workflow steps."""
    try:
        components = deepcopy(components)
        with preserve_contract_schemas(components):
            for source, destination in zip(components, components[1:], strict=False):
                _validate_sequential_contract(source, destination)

            flow_outputs = deepcopy(public_outputs(components[-1]))
            start_node = StartNode(
                name="sequential-start",
                inputs=list(components[0].inputs or ()),
            )
            step_nodes = [
                _make_sequential_step_node(component, position=position)
                for position, component in enumerate(components)
            ]
            end_node = _end_node(
                name="sequential-end",
                outputs=public_outputs(components[-1]),
            )
            nodes = [start_node, *step_nodes, end_node]

            return guarded_flow(
                outputs=flow_outputs,
                name=name,
                start_node=start_node,
                nodes=nodes,
                control_flow_connections=_control_edges(nodes),
                data_flow_connections=_data_edges(nodes),
            )
    except PatternCompilationError:
        raise
    except Exception:
        raise PatternCompilationError(
            "Sequential could not be compiled into an AgentSpec Flow."
        ) from None


def _validate_parallel_contract(branches: Sequence[RuntimeComponent]) -> None:
    """Require compatible shared inputs and globally unique branch output names."""
    inputs: dict[str, tuple[Property, str]] = {}
    outputs: dict[str, str] = {}

    for branch in branches:
        branch_inputs: set[str] = set()
        for property_ in branch.inputs or ():
            if property_.title in branch_inputs:
                raise IncompatibleParallelContractError(
                    f"Parallel branch '{branch.name}' has duplicate input property "
                    f"'{property_.title}'."
                )
            branch_inputs.add(property_.title)

            existing_input = inputs.get(property_.title)
            if existing_input is None:
                inputs[property_.title] = (property_, branch.name)
            elif not properties_have_same_type(existing_input[0], property_):
                raise IncompatibleParallelContractError(
                    f"Parallel branches '{existing_input[1]}' and '{branch.name}' use input "
                    f"'{property_.title}' with incompatible types."
                )

        for property_ in public_outputs(branch):
            existing_branch = outputs.get(property_.title)
            if existing_branch is not None:
                raise IncompatibleParallelContractError(
                    f"Parallel branches '{existing_branch}' and '{branch.name}' have "
                    f"duplicate output property '{property_.title}'."
                )
            outputs[property_.title] = branch.name


def _build_parallel_branch_flow(
    component: RuntimeComponent,
    *,
    position: int,
) -> Flow:
    """Adapt an agent or node to a subflow while preserving its external contract."""
    start_node = StartNode(
        name=f"parallel-branch-{position:03d}-start",
        inputs=list(component.inputs or ()),
    )
    component_node_name = f"parallel-branch-{position:03d}-{component.name}"
    if isinstance(component, Agent):
        component_node: Node = AgentNode(
            name=component_node_name,
            agent=component,
        )
    elif isinstance(component, Node):
        component_node = component.model_copy(
            update={"id": str(uuid4()), "name": component_node_name},
        )
    else:
        raise PatternCompilationError(f"Parallel component '{component.name}' is not supported.")
    end_node = _end_node(
        name=f"parallel-branch-{position:03d}-end",
        outputs=list(component.outputs or ()),
    )
    nodes = [start_node, component_node, end_node]

    return Flow(
        name=f"parallel-branch-{position:03d}-{component.name}",
        start_node=start_node,
        nodes=nodes,
        control_flow_connections=_control_edges(nodes),
        data_flow_connections=_data_edges(nodes),
    )


def build_parallel_flow(branches: Sequence[RuntimeComponent], *, name: str) -> Flow:
    """Validate and combine independent workflow branches."""
    try:
        branches = deepcopy(branches)
        with preserve_contract_schemas(branches):
            _validate_parallel_contract(branches)
            flow_outputs = deepcopy(
                [prop for branch in branches for prop in public_outputs(branch)]
            )
            subflows = [
                branch
                if isinstance(branch, Flow)
                else _build_parallel_branch_flow(branch, position=position)
                for position, branch in enumerate(branches)
            ]
            parallel_node = ParallelFlowNode(name="parallel-branches", subflows=subflows)

            start_node = StartNode(
                name="parallel-start",
                inputs=list(parallel_node.inputs or ()),
            )
            end_node = _end_node(
                name="parallel-end",
                outputs=public_outputs(parallel_node),
            )
            nodes = [start_node, parallel_node, end_node]

            return guarded_flow(
                outputs=flow_outputs,
                name=name,
                start_node=start_node,
                nodes=nodes,
                control_flow_connections=_control_edges(nodes),
                data_flow_connections=_data_edges(nodes),
            )
    except PatternCompilationError:
        raise
    except Exception:
        raise PatternCompilationError(
            "Parallel could not be compiled into an AgentSpec Flow."
        ) from None


def _routing_properties_by_title(
    component: RuntimeComponent,
    *,
    direction: str,
) -> dict[str, Property]:
    """Index routing inputs or outputs by title, rejecting duplicate names."""
    properties = component.inputs if direction == "input" else public_outputs(component)
    properties_by_title: dict[str, Property] = {}
    duplicates: list[str] = []
    for property_ in properties or ():
        if property_.title in properties_by_title:
            duplicates.append(property_.title)
        properties_by_title[property_.title] = property_
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise IncompatibleRoutingContractError(
            f"Routing component '{component.name}' has duplicate {direction} "
            f"properties: {duplicate_list}."
        )
    return properties_by_title


def _validate_routing_contract(
    router: RuntimeComponent,
    *,
    route_key: str,
    destinations: Sequence[tuple[str, RuntimeComponent]],
) -> tuple[Property, list[Property]]:
    """Validate routing data flow and return its route property and shared outputs.

    Destinations consume a compatible subset of the router's original inputs,
    not the router's outputs. All destinations need matching output contracts.
    """
    router_inputs = _routing_properties_by_title(router, direction="input")
    router_outputs = _routing_properties_by_title(router, direction="output")

    route_property = router_outputs.get(route_key)
    if route_property is None:
        raise IncompatibleRoutingContractError(
            f"Routing router '{router.name}' does not expose route key '{route_key}'."
        )
    if route_property.json_schema.get("type") != "string":
        raise IncompatibleRoutingContractError(
            f"Routing route key '{route_key}' must be a string property."
        )

    destination_outputs: list[tuple[str, dict[str, Property]]] = []
    for label, destination in destinations:
        inputs = _routing_properties_by_title(destination, direction="input")
        missing_inputs = sorted(set(inputs) - set(router_inputs))
        if missing_inputs:
            raise IncompatibleRoutingContractError(
                f"Routing destination '{destination.name}' for '{label}' requires "
                f"inputs not provided by the routing workflow input contract: "
                f"{missing_inputs}."
            )

        incompatible_inputs = [
            title
            for title in sorted(inputs)
            if not properties_have_same_type(router_inputs[title], inputs[title])
        ]
        if incompatible_inputs:
            raise IncompatibleRoutingContractError(
                f"Routing destination '{destination.name}' for '{label}' has "
                f"incompatible input types for {incompatible_inputs}."
            )

        destination_outputs.append(
            (
                label,
                _routing_properties_by_title(destination, direction="output"),
            )
        )

    reference_label, reference_outputs = destination_outputs[0]
    for label, outputs in destination_outputs[1:]:
        reference_names = set(reference_outputs)
        output_names = set(outputs)
        if output_names != reference_names:
            missing = sorted(reference_names - output_names)
            additional = sorted(output_names - reference_names)
            details: list[str] = []
            if missing:
                details.append(f"missing outputs {missing}")
            if additional:
                details.append(f"additional outputs {additional}")
            raise IncompatibleRoutingContractError(
                f"Routing destination '{label}' is incompatible with "
                f"'{reference_label}': " + "; ".join(details) + "."
            )

        incompatible_outputs = [
            title
            for title in sorted(reference_names)
            if not properties_have_same_type(reference_outputs[title], outputs[title])
        ]
        if incompatible_outputs:
            raise IncompatibleRoutingContractError(
                f"Routing destination '{label}' is incompatible with "
                f"'{reference_label}': incompatible output types for "
                f"{incompatible_outputs}."
            )

    common_outputs = [reference_outputs[title] for title in sorted(reference_outputs)]
    return route_property, common_outputs


def _make_routing_node(component: RuntimeComponent, *, name: str) -> Node:
    """Adapt a routing component to a named node without changing the original."""
    if isinstance(component, Flow):
        return FlowNode(name=name, subflow=component)
    if isinstance(component, Agent):
        return AgentNode(name=name, agent=component)
    if isinstance(component, Node):
        return component.model_copy(update={"id": str(uuid4()), "name": name})
    raise PatternCompilationError(f"Routing component '{component.name}' is not supported.")


def _routing_transition_edges(
    source: Node,
    destination: Node,
    *,
    name_prefix: str,
) -> list[ControlFlowEdge]:
    """Connect every source exit to the next node within a routing path."""
    edges: list[ControlFlowEdge] = []
    for position, branch in enumerate(sorted(source.branches or [Node.DEFAULT_NEXT_BRANCH])):
        edges.append(
            ControlFlowEdge(
                name=f"{name_prefix}-{position:03d}",
                from_node=source,
                from_branch=(
                    None
                    if branch == Node.DEFAULT_NEXT_BRANCH and len(source.branches or []) <= 1
                    else branch
                ),
                to_node=destination,
            )
        )
    return edges


def build_routing_flow(
    router: RuntimeComponent,
    *,
    name: str,
    route_key: str,
    routes: Mapping[str, RuntimeComponent],
    default: RuntimeComponent,
) -> Flow:
    """Build a router followed by one selected destination or the default path.

    Sort route labels for deterministic graph construction. Feed the routing
    decision from the router and destination data from the original flow inputs.
    """
    try:
        router, routes, default = deepcopy((router, routes, default))
        with preserve_contract_schemas(router, routes, default):
            ordered_routes = sorted(routes.items())
            destinations = [*ordered_routes, ("default", default)]
            route_property, common_outputs = _validate_routing_contract(
                router,
                route_key=route_key,
                destinations=destinations,
            )

            flow_outputs = deepcopy(common_outputs)
            start_node = StartNode(
                name="routing-start",
                inputs=list(router.inputs or ()),
            )
            router_node = _make_routing_node(
                router,
                name=f"routing-router-{router.name}",
            )
            route_branches = {
                value: f"routing-route-{position:03d}"
                for position, (value, _) in enumerate(ordered_routes)
            }
            branching_node = BranchingNode(
                name="routing-branch",
                mapping=route_branches,
                inputs=[route_property],
            )

            target_nodes: list[Node] = []
            end_nodes: list[EndNode] = []
            route_paths: list[tuple[str, str, Node, EndNode]] = []
            for position, (value, destination) in enumerate(ordered_routes):
                branch_name = route_branches[value]
                target_node = _make_routing_node(
                    destination,
                    name=f"routing-target-{position:03d}-{destination.name}",
                )
                end_node = _end_node(
                    name=f"routing-end-{position:03d}",
                    branch_name=branch_name,
                    outputs=list(common_outputs),
                )
                target_nodes.append(target_node)
                end_nodes.append(end_node)
                route_paths.append((value, branch_name, target_node, end_node))

            default_node = _make_routing_node(
                default,
                name=f"routing-default-{default.name}",
            )
            default_end = _end_node(
                name="routing-default-end",
                branch_name=BranchingNode.DEFAULT_BRANCH,
                outputs=list(common_outputs),
            )
            target_nodes.append(default_node)
            end_nodes.append(default_end)

            control_edges = [
                *_routing_transition_edges(
                    start_node,
                    router_node,
                    name_prefix="routing-control-start-router",
                ),
                *_routing_transition_edges(
                    router_node,
                    branching_node,
                    name_prefix="routing-control-router-branch",
                ),
            ]
            for position, (_, branch_name, target_node, end_node) in enumerate(route_paths):
                control_edges.append(
                    ControlFlowEdge(
                        name=f"routing-control-branch-target-{position:03d}",
                        from_node=branching_node,
                        from_branch=branch_name,
                        to_node=target_node,
                    )
                )
                control_edges.extend(
                    _routing_transition_edges(
                        target_node,
                        end_node,
                        name_prefix=f"routing-control-target-end-{position:03d}",
                    )
                )
            control_edges.append(
                ControlFlowEdge(
                    name="routing-control-branch-default",
                    from_node=branching_node,
                    from_branch=BranchingNode.DEFAULT_BRANCH,
                    to_node=default_node,
                )
            )
            control_edges.extend(
                _routing_transition_edges(
                    default_node,
                    default_end,
                    name_prefix="routing-control-default-end",
                )
            )

            data_edges = [
                DataFlowEdge(
                    name=f"routing-data-start-router-{property_.title}",
                    source_node=start_node,
                    source_output=property_.title,
                    destination_node=router_node,
                    destination_input=property_.title,
                )
                for property_ in sorted(
                    router.inputs or (),
                    key=lambda property_: property_.title,
                )
            ]
            data_edges.append(
                DataFlowEdge(
                    name=f"routing-data-router-branch-{route_key}",
                    source_node=router_node,
                    source_output=route_key,
                    destination_node=branching_node,
                    destination_input=route_key,
                )
            )

            all_paths = [
                *[
                    (position, target_node, end_node)
                    for position, (_, _, target_node, end_node) in enumerate(route_paths)
                ],
                (len(route_paths), default_node, default_end),
            ]
            for position, target_node, end_node in all_paths:
                for property_ in sorted(
                    target_node.inputs or (),
                    key=lambda property_: property_.title,
                ):
                    data_edges.append(
                        DataFlowEdge(
                            name=(f"routing-data-start-target-{position:03d}-{property_.title}"),
                            source_node=start_node,
                            source_output=property_.title,
                            destination_node=target_node,
                            destination_input=property_.title,
                        )
                    )
                for property_ in common_outputs:
                    data_edges.append(
                        DataFlowEdge(
                            name=(f"routing-data-target-end-{position:03d}-{property_.title}"),
                            source_node=target_node,
                            source_output=property_.title,
                            destination_node=end_node,
                            destination_input=property_.title,
                        )
                    )

            return guarded_flow(
                outputs=flow_outputs,
                name=name,
                start_node=start_node,
                nodes=[
                    start_node,
                    router_node,
                    branching_node,
                    *target_nodes,
                    *end_nodes,
                ],
                control_flow_connections=control_edges,
                data_flow_connections=data_edges,
            )
    except PatternCompilationError:
        raise
    except Exception:
        raise PatternCompilationError(
            "Routing could not be compiled into an AgentSpec Flow."
        ) from None
