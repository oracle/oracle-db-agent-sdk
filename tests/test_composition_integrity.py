##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Regression checks for graph identity and contract preservation during composition."""

import json
from copy import deepcopy

import pytest
from pyagentspec import AgentSpecDeserializer

from oracle_db_agent_sdk import (
    IncompatibleSequentialContractError,
    agent,
    agent_tool,
    llm_node,
    loop,
    orchestrator,
    parallel,
    plsql_tool,
    routing,
    sequential,
    tool_node,
)
from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import _unwrap_executable, _unwrap_tool


def make_step(llm, kind, *, name="step", inputs=None, outputs=None):
    inputs = {"value": "string"} if inputs is None else inputs
    outputs = inputs if outputs is None else outputs
    prompt = "Process " + " ".join("{{" + title + "}}" for title in inputs)
    if kind == "llm":
        return llm_node(name=name, llm=llm, prompt=prompt, inputs=inputs, outputs=outputs)
    if kind == "agent":
        return agent(name=name, llm=llm, instruction=prompt, inputs=inputs, outputs=outputs)
    return tool_node(
        name=name,
        tool=plsql_tool(name=name, procedure="EXAMPLE", inputs=inputs, outputs=outputs),
    )


@pytest.mark.parametrize("kind", ["llm", "tool", "agent"])
def test_reused_steps_keep_separate_control_and_data_paths(database_llm, kind) -> None:
    step = make_step(database_llm, kind)
    original = deepcopy(_unwrap_executable(step))
    workflow = sequential(name="twice", steps=[step, step])

    graph = AgentSpecDeserializer().from_json(_serialize_workflow(workflow))

    assert len({node.id for node in graph.nodes}) == 4
    assert [(edge.from_node.id, edge.to_node.id) for edge in graph.control_flow_connections] == [
        (source.id, destination.id)
        for source, destination in zip(graph.nodes, graph.nodes[1:], strict=False)
    ]
    assert [
        (edge.source_node.id, edge.source_output, edge.destination_node.id, edge.destination_input)
        for edge in graph.data_flow_connections
    ] == [
        (source.id, "value", destination.id, "value")
        for source, destination in zip(graph.nodes, graph.nodes[1:], strict=False)
    ]
    assert _unwrap_executable(step) == original


@pytest.mark.parametrize("kind", ["llm", "tool", "agent"])
def test_reused_routing_targets_keep_each_route_and_default_distinct(database_llm, kind) -> None:
    target = make_step(database_llm, kind)
    router = make_step(database_llm, "llm", name="router", outputs={"route": "string"})
    workflow = routing(
        router,
        name="routing_test",
        route_key="route",
        routes={"a": target, "b": target},
        default=target,
    )

    graph = AgentSpecDeserializer().from_json(_serialize_workflow(workflow))
    branch_node = graph.nodes[2]
    branch_edges = [
        edge for edge in graph.control_flow_connections if edge.from_node.id == branch_node.id
    ]

    assert len({node.id for node in graph.nodes}) == len(graph.nodes)
    assert len(branch_edges) == 3
    assert len({edge.to_node.id for edge in branch_edges}) == 3
    targets = {edge.to_node.id for edge in branch_edges}
    assert {
        edge.destination_node.id
        for edge in graph.data_flow_connections
        if edge.source_node.id == graph.start_node.id and edge.destination_node.id in targets
    } == targets
    assert len({node.name for node in graph.nodes}) == len(graph.nodes)


def test_reused_parallel_tool_nodes_get_distinct_ids(database_llm) -> None:
    # Empty tool outputs allow reuse without conflicting parallel output names.
    step = make_step(database_llm, "tool", outputs={})
    graph = AgentSpecDeserializer().from_json(
        _serialize_workflow(parallel(name="parallel_test", steps=[step, step]))
    )
    first, second = graph.nodes[1].subflows

    assert first.nodes[1].id != second.nodes[1].id
    assert first.nodes[1].name != second.nodes[1].name
    assert first.control_flow_connections[0].to_node.id == first.nodes[1].id
    assert second.control_flow_connections[0].to_node.id == second.nodes[1].id


STRING_SCHEMA = {"type": "string", "anyOf": [{"type": "string", "maxLength": 10}]}
SCHEMAS = [
    STRING_SCHEMA,
    {**STRING_SCHEMA, "default": "ok"},
    {"type": "array", "items": STRING_SCHEMA},
    {"type": "object", "properties": {"text": STRING_SCHEMA}},
]


def compose(pattern, llm, step, contract):
    if pattern == "standalone":
        return step
    if pattern == "sequential":
        return sequential(name="twice", steps=[step, step])
    if pattern == "parallel":
        other = make_step(llm, "agent", name="other", inputs=contract, outputs={"other": "string"})
        return parallel(name="parallel_test", steps=[step, other])
    if pattern == "routing":
        router = make_step(llm, "llm", name="router", inputs=contract, outputs={"route": "string"})
        return routing(
            router, name="routing_test", route_key="route", routes={"a": step}, default=step
        )
    if pattern == "nested":
        inner = sequential(name="inner", steps=[step, step])
        return sequential(name="outer", steps=[inner, inner])
    return loop(name="repeat", body=step, state_key="done", continue_value="N", max_iterations=2)


def value_schemas(document):
    """Inspect the actual wire document, including a loop's embedded body."""
    if isinstance(document, dict):
        if document.get("title") == "value":
            yield document
        if document.get("title") == "body_spec":
            yield from value_schemas(json.loads(document["default"]))
        for key, value in document.items():
            if (
                key == "outputs"
                and document.get("component_type") == "Flow"
                and document.get("metadata", {}).get("oracle_db_agent_sdk_loop_status")
            ):
                # Private transport defaults support an exhaustion-only exit.
                # Original public contracts and successful EndNodes are checked below.
                continue
            yield from value_schemas(value)
    elif isinstance(document, list):
        for value in document:
            yield from value_schemas(value)


@pytest.mark.parametrize("schema", SCHEMAS, ids=["union", "default", "array", "object"])
@pytest.mark.parametrize("kind", ["llm", "tool", "agent"])
@pytest.mark.parametrize(
    "pattern", ["standalone", "sequential", "parallel", "routing", "nested", "loop"]
)
def test_composition_preserves_exact_source_and_serialized_schemas(
    database_llm, schema, kind, pattern
) -> None:
    contract = {"done": "string", "value": deepcopy(schema)}
    original_contract = deepcopy(contract)
    step = make_step(database_llm, kind, inputs=contract)
    original_step = deepcopy(_unwrap_executable(step))
    expected = {**schema, "title": "value"}

    workflow = compose(pattern, database_llm, step, contract)
    schemas = [
        schema
        for spec in [_serialize_workflow(workflow)]
        for schema in value_schemas(json.loads(spec))
    ]

    assert schemas
    assert all(value == expected for value in schemas)
    assert contract == original_contract
    assert _unwrap_executable(step) == original_step


@pytest.mark.parametrize("pattern", ["sequential", "parallel", "routing", "loop"])
def test_repeated_compilation_does_not_accumulate_union_members(database_llm, pattern) -> None:
    contract = {"done": "string", "value": deepcopy(STRING_SCHEMA)}
    step = make_step(database_llm, "agent", inputs=contract)
    before = deepcopy(_unwrap_executable(step))

    for _ in range(60):
        workflow = compose(pattern, database_llm, step, contract)

    assert _unwrap_executable(step) == before
    schemas = [
        schema
        for spec in [_serialize_workflow(workflow)]
        for schema in value_schemas(json.loads(spec))
    ]
    assert schemas
    assert all(value == {**STRING_SCHEMA, "title": "value"} for value in schemas)


def test_failed_compilation_preserves_reusable_contracts(database_llm) -> None:
    first = make_step(database_llm, "agent", inputs={"value": deepcopy(STRING_SCHEMA)})
    second = make_step(database_llm, "agent", name="other", inputs={"value": "integer"})
    before = deepcopy(_unwrap_executable(first))

    with pytest.raises(IncompatibleSequentialContractError):
        sequential(name="invalid", steps=[first, second])

    assert _unwrap_executable(first) == before


@pytest.mark.parametrize(
    ("tool_kind", "factory"),
    [("plsql", agent), ("plsql", orchestrator), ("agent", orchestrator)],
)
def test_tools_with_defaults_retain_schemas_when_shared_with_agents(
    database_llm, tool_kind, factory
) -> None:
    schema = {**STRING_SCHEMA, "default": "ok"}
    arguments = dict(name="capability", inputs={"value": schema}, outputs={"value": schema})
    if tool_kind == "plsql":
        tool = plsql_tool(procedure="EXAMPLE", **arguments)
    else:
        tool = agent_tool(agent_name="EXAMPLE", schema="APP", **arguments)
    before = deepcopy(_unwrap_tool(tool))

    for _ in range(3):
        workflow = factory(
            name="user",
            llm=database_llm,
            instruction="Handle {{request}}.",
            inputs={"request": "string"},
            outputs={"answer": "string"},
            tools=[tool],
        )
        schemas = [
            schema
            for spec in [_serialize_workflow(workflow)]
            for schema in value_schemas(json.loads(spec))
        ]
        assert schemas
        assert all(value == {**schema, "title": "value"} for value in schemas)

    assert _unwrap_tool(tool) == before
