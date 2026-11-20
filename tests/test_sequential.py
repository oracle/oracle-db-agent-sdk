##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for the public sequential authoring pattern."""

import pytest
from pyagentspec import Agent
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import (
    AgentNode,
    EndNode,
    FlowNode,
    LlmNode,
    StartNode,
    ToolNode,
)
from pyagentspec.property import Property

from oracle_db_agent_sdk import (
    IncompatibleSequentialContractError,
    PatternCompilationError,
    llm_node,
    plsql_tool,
    sequential,
    tool_node,
)
from oracle_db_agent_sdk._wrappers import Executable, LlmConfig, _unwrap_executable


def _unwrap_flow(value: Executable) -> Flow:
    component = _unwrap_executable(value)
    assert isinstance(component, Flow)
    return component


def _unwrap_agent(value: Executable) -> Agent:
    component = _unwrap_executable(value)
    assert isinstance(component, Agent)
    return component


def test_sequential_builds_standard_flow_nodes(make_agent) -> None:
    agents = [make_agent(f"a{index}") for index in range(1, 4)]

    flow = _unwrap_flow(sequential(name="three_step_sequence", steps=agents))
    raw_agents = [_unwrap_agent(value) for value in agents]

    assert [type(node) for node in flow.nodes] == [
        StartNode,
        AgentNode,
        AgentNode,
        AgentNode,
        EndNode,
    ]
    assert [node.agent for node in flow.nodes[1:-1]] == raw_agents
    assert len(flow.control_flow_connections) == 4
    assert [(edge.from_node.name, edge.to_node.name) for edge in flow.control_flow_connections] == [
        (source.name, destination.name)
        for source, destination in zip(flow.nodes, flow.nodes[1:], strict=False)
    ]
    assert [
        (
            edge.source_node.name,
            edge.source_output,
            edge.destination_node.name,
            edge.destination_input,
        )
        for edge in flow.data_flow_connections or []
    ] == [
        (
            source.name,
            "value",
            destination.name,
            "value",
        )
        for source, destination in zip(flow.nodes, flow.nodes[1:], strict=False)
    ]


def test_sequential_supports_named_step_list(make_agent) -> None:
    agents = [make_agent("a1"), make_agent("a2")]

    flow = _unwrap_flow(sequential(name="diagnostic_sequence", steps=agents))

    assert flow.name == "diagnostic_sequence"
    assert [node.agent for node in flow.nodes[1:-1]] == [_unwrap_agent(value) for value in agents]


def test_sequential_composes_tool_and_llm_nodes(database_llm: LlmConfig) -> None:
    lookup = tool_node(
        name="lookup",
        tool=plsql_tool(
            name="lookup_status",
            procedure="ORDER_SERVICE.GET_STATUS",
            inputs={"order_id": "string"},
            outputs={"order_status": "string"},
        ),
    )
    explain = llm_node(
        name="explain",
        llm=database_llm,
        prompt="Explain {{order_status}}.",
        inputs={"order_status": "string"},
        outputs={"answer": "string"},
    )

    flow = _unwrap_flow(sequential(name="order_support", steps=[lookup, explain]))

    assert [type(node) for node in flow.nodes] == [
        StartNode,
        ToolNode,
        LlmNode,
        EndNode,
    ]
    assert [property_.title for property_ in flow.inputs or ()] == ["order_id"]
    assert [property_.title for property_ in flow.outputs or ()] == ["answer"]


@pytest.mark.parametrize(
    ("name", "steps", "message"),
    [
        ("", [], "name"),
        ("sequence", [], "at least two"),
        ("sequence", [object(), object()], "oracle_db_agent_sdk building block"),
    ],
)
def test_sequential_rejects_invalid_arguments(
    name,
    steps,
    message,
) -> None:
    with pytest.raises(PatternCompilationError, match=message):
        sequential(name=name, steps=steps)


def test_construction_does_not_mutate_source_components(make_agent) -> None:
    a1, a2 = make_agent("a1"), make_agent("a2")
    raw_agents = [_unwrap_agent(a1), _unwrap_agent(a2)]
    before = [value.model_copy(deep=True) for value in raw_agents]

    sequential(name="sequence", steps=[a1, a2])

    assert raw_agents == before


def test_existing_flow_is_wrapped_in_flow_node(make_agent) -> None:
    a1, a2, a3 = make_agent("a1"), make_agent("a2"), make_agent("a3")
    existing_workflow = sequential(name="inner_sequence", steps=[a1, a2])
    existing_flow = _unwrap_flow(existing_workflow)

    outer_flow = _unwrap_flow(sequential(name="outer_sequence", steps=[existing_workflow, a3]))

    assert isinstance(outer_flow.nodes[1], FlowNode)
    assert outer_flow.nodes[1].subflow == existing_flow
    assert outer_flow.nodes[1].subflow is not existing_flow
    assert isinstance(outer_flow.nodes[2], AgentNode)


@pytest.mark.parametrize(
    ("source_outputs", "destination_inputs", "message"),
    [
        ((("other", "string"),), (("value", "string"),), "missing outputs"),
        (
            (("value", "string"), ("extra", "string")),
            (("value", "string"),),
            "additional outputs",
        ),
        ((("value", "integer"),), (("value", "string"),), "incompatible property"),
    ],
)
def test_incompatible_adjacent_contract_fails(
    make_agent,
    source_outputs,
    destination_inputs,
    message,
) -> None:
    source = make_agent("source", outputs=source_outputs)
    destination = make_agent("destination", inputs=destination_inputs)

    with pytest.raises(IncompatibleSequentialContractError, match=message):
        sequential(name="sequence", steps=[source, destination])


def test_duplicate_output_names_fail_clearly(make_agent) -> None:
    source = make_agent("source")
    duplicate = Property(json_schema={"title": "value", "type": "string"})
    _unwrap_agent(source).outputs = [duplicate, duplicate.model_copy()]
    destination = make_agent("destination")

    with pytest.raises(IncompatibleSequentialContractError, match="duplicate output"):
        sequential(name="sequence", steps=[source, destination])
