##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for the internal serialization boundary."""

import json
import traceback
from copy import deepcopy

import pytest
from pyagentspec import AgentSpecDeserializer, AgentSpecSerializer
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import EndNode, StartNode, ToolNode

import oracle_db_agent_sdk._pattern_compiler as compiler_module
import oracle_db_agent_sdk._serialization as serialization_module
import oracle_db_agent_sdk.executor as executor_module
from oracle_db_agent_sdk import (
    AgentSpecSerializationError,
    BuildingBlockError,
    PatternCompilationError,
    agent,
    execute,
    llm_config,
    llm_node,
    loop,
    parallel,
    plsql_tool,
    routing,
    sequential,
    tool_node,
)
from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import _unwrap_executable


def test_direct_component_uses_official_serializer_once(make_agent, monkeypatch) -> None:
    agent = make_agent("agent")
    captured = []

    class FakeSerializer:
        def to_json(self, component):
            captured.append(component)
            return '{"component_type":"Agent"}'

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FakeSerializer)

    assert _serialize_workflow(agent) == '{"component_type":"Agent"}'
    assert captured == [_unwrap_executable(agent)]


def test_empty_agent_transforms_are_omitted_without_rewriting_application_data(
    make_agent,
    monkeypatch,
):
    application_value = {"component_type": "Agent", "transforms": []}
    wire = {
        "component_type": "Agent",
        "transforms": [],
        "inputs": [{"title": "data", "type": "object", "default": application_value}],
        "metadata": {"example": application_value},
        "configuration": {"example": application_value},
        "$referenced_components": {"child": {"component_type": "Agent", "transforms": []}},
        "tools": [{"component_type": "Agent", "transforms": ["keep"]}],
    }

    class FakeSerializer:
        def to_json(self, component):
            return json.dumps(wire)

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FakeSerializer)
    result = json.loads(_serialize_workflow(make_agent("test")))
    assert "transforms" not in result
    assert "transforms" not in result["$referenced_components"]["child"]
    assert result["tools"][0]["transforms"] == ["keep"]
    assert result["inputs"] == wire["inputs"]
    assert result["metadata"] == wire["metadata"]
    assert result["configuration"] == wire["configuration"]


def test_standalone_node_is_wrapped_in_standard_flow(monkeypatch) -> None:
    workflow = tool_node(
        name="lookup_order",
        tool=plsql_tool(
            name="lookup_order",
            procedure="LOOKUP_ORDER",
            inputs={"order_id": "string"},
            outputs={"status": "string"},
        ),
    )
    original_node = _unwrap_executable(workflow)
    captured = []

    class FakeSerializer:
        def to_json(self, component):
            captured.append(component)
            return "{}"

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FakeSerializer)

    _serialize_workflow(workflow)

    assert len(captured) == 1
    flow = captured[0]
    assert isinstance(flow, Flow)
    assert flow.name == "lookup_order-workflow"
    assert [type(node) for node in flow.nodes] == [StartNode, ToolNode, EndNode]
    assert flow.nodes[1] == original_node
    assert flow.nodes[1] is not original_node
    assert [(edge.from_node.name, edge.to_node.name) for edge in flow.control_flow_connections] == [
        ("standalone-start", "lookup_order"),
        ("lookup_order", "standalone-end"),
    ]
    assert [
        (
            edge.source_node.name,
            edge.source_output,
            edge.destination_node.name,
            edge.destination_input,
        )
        for edge in flow.data_flow_connections or ()
    ] == [
        ("standalone-start", "order_id", "lookup_order", "order_id"),
        ("lookup_order", "status", "standalone-end", "status"),
    ]


def test_eager_sequential_flow_uses_official_serializer(make_agent, monkeypatch) -> None:
    workflow = sequential(
        name="sequence",
        steps=[make_agent("a1"), make_agent("a2")],
    )
    flow = _unwrap_executable(workflow)
    captured = []

    class FakeSerializer:
        def to_json(self, component):
            captured.append(component)
            return "{}"

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FakeSerializer)

    _serialize_workflow(workflow)

    assert len(captured) == 1
    assert captured == [flow]
    assert isinstance(flow, Flow)


def test_eager_routing_flow_uses_official_serializer(make_agent, monkeypatch) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    general = make_agent("general")
    workflow = routing(
        router,
        name="routing_test",
        route_key="issue_type",
        routes={"general": general},
        default=general,
    )
    flow = _unwrap_executable(workflow)
    captured = []

    class FakeSerializer:
        def to_json(self, component):
            captured.append(component)
            return "{}"

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FakeSerializer)

    _serialize_workflow(workflow)

    assert captured == [flow]
    assert isinstance(flow, Flow)


def test_serialized_pattern_contains_only_standard_components(make_agent) -> None:
    nested_sequence = sequential(
        name="nested_sequence",
        steps=[make_agent("a2"), make_agent("a3")],
    )
    pattern = sequential(
        name="outer_sequence",
        steps=[make_agent("a1"), nested_sequence],
    )

    document = json.loads(_serialize_workflow(pattern))

    def component_types(value):
        if isinstance(value, dict):
            if "component_type" in value:
                yield value["component_type"]
            for nested in value.values():
                yield from component_types(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from component_types(nested)

    types = list(component_types(document))
    assert document["component_type"] == "Flow"
    assert "Sequential" not in types
    assert {"StartNode", "AgentNode", "EndNode", "ControlFlowEdge", "DataFlowEdge"} <= set(types)


def test_serialized_parallel_contains_only_standard_components(make_agent) -> None:
    pattern = parallel(
        name="parallel_test",
        steps=[
            make_agent(
                "medical",
                outputs=(("medical_result", "string"),),
            ),
            make_agent(
                "fraud",
                outputs=(("fraud_result", "string"),),
            ),
        ],
    )

    document = json.loads(_serialize_workflow(pattern))

    def component_types(value):
        if isinstance(value, dict):
            if "component_type" in value:
                yield value["component_type"]
            for nested in value.values():
                yield from component_types(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from component_types(nested)

    types = list(component_types(document))
    assert document["component_type"] == "Flow"
    assert "Parallel" not in types
    assert "ParallelFlowNode" in types
    assert {"StartNode", "AgentNode", "EndNode", "ControlFlowEdge", "DataFlowEdge"} <= set(types)


def test_serialized_routing_contains_only_standard_components(make_agent) -> None:
    router = make_agent(
        "router",
        outputs=(("issue_type", "string"),),
    )
    billing = make_agent(
        "billing",
        outputs=(("resolution", "string"),),
    )
    general = make_agent(
        "general",
        outputs=(("resolution", "string"),),
    )
    pattern = routing(
        router,
        name="routing_test",
        route_key="issue_type",
        routes={"billing": billing},
        default=general,
    )

    document = json.loads(_serialize_workflow(pattern))

    def component_types(value):
        if isinstance(value, dict):
            if "component_type" in value:
                yield value["component_type"]
            for nested in value.values():
                yield from component_types(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from component_types(nested)

    types = list(component_types(document))
    assert document["component_type"] == "Flow"
    assert "Routing" not in types
    assert "BranchingNode" in types
    assert {"StartNode", "AgentNode", "EndNode", "ControlFlowEdge", "DataFlowEdge"} <= set(types)


def test_serializer_failure_is_wrapped_without_payload(make_agent, monkeypatch) -> None:
    original = RuntimeError("synthetic-sensitive-value")

    class FailingSerializer:
        def to_json(self, component):
            raise original

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FailingSerializer)

    with pytest.raises(AgentSpecSerializationError) as caught:
        _serialize_workflow(make_agent("agent"))

    assert caught.value.__cause__ is None
    assert "synthetic-sensitive-value" not in str(caught.value)
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("pattern", ["standalone", "sequential", "parallel", "routing"])
def test_compilation_failure_is_wrapped_without_payload(make_agent, monkeypatch, pattern) -> None:
    original = RuntimeError("synthetic-sensitive-value")
    first = make_agent("first")
    second = make_agent("second", outputs=(("answer", "string"),))
    router = make_agent("router", outputs=(("route", "string"),))
    node = tool_node(
        name="lookup",
        tool=plsql_tool(name="lookup", procedure="LOOKUP", inputs={}, outputs={}),
    )
    build = {
        "standalone": lambda: _serialize_workflow(node),
        "sequential": lambda: sequential(name="sequence", steps=[first, second]),
        "parallel": lambda: parallel(name="parallel_test", steps=[first, second]),
        "routing": lambda: routing(
            router, name="routing_test", route_key="route", routes={"first": first}, default=first
        ),
    }

    def fail_start_node(*args, **kwargs):
        raise original

    monkeypatch.setattr(compiler_module, "StartNode", fail_start_node)

    with pytest.raises(PatternCompilationError) as caught:
        build[pattern]()

    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None


def test_serializer_rejects_raw_pyagentspec_component(make_agent) -> None:
    raw_agent = _unwrap_executable(make_agent("agent"))

    with pytest.raises(BuildingBlockError, match="workflow must be created"):
        _serialize_workflow(raw_agent)  # type: ignore[arg-type]


def test_standalone_compilation_error_is_not_reclassified(monkeypatch) -> None:
    workflow = tool_node(
        name="lookup_order",
        tool=plsql_tool(
            name="lookup_order",
            procedure="LOOKUP_ORDER",
            inputs={"order_id": "string"},
            outputs={"status": "string"},
        ),
    )
    original = PatternCompilationError("standalone compilation failed")

    def fail_compilation(node):
        raise original

    monkeypatch.setattr(
        serialization_module,
        "build_standalone_node_flow",
        fail_compilation,
    )

    with pytest.raises(PatternCompilationError) as caught:
        _serialize_workflow(workflow)

    assert caught.value is original


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("location", ["default", "nested-default", "generation"])
def test_nonfinite_workflow_numbers_are_rejected_before_database(
    monkeypatch, number, location
) -> None:
    def unexpected_database_call(*args, **kwargs):
        pytest.fail("Invalid AgentSpec JSON reached the database")

    monkeypatch.setattr(executor_module, "execute_immediate", unexpected_database_call)
    if location == "generation":
        config = llm_config(
            provider="openai",
            model_id="example-model",
            url="https://example.invalid/chat",
            credential_name="TEST_CREDENTIAL",
            generation={"temperature": number},
        )
        workflow = agent(name="step", llm=config, instruction="Do work", inputs={}, outputs={})
    else:
        schema = (
            {"type": "object", "default": {"numbers": [number]}}
            if location == "nested-default"
            else {"type": "number", "default": number}
        )
        workflow = tool_node(
            name="step",
            tool=plsql_tool(name="tool", procedure="EXAMPLE", inputs={"value": schema}, outputs={}),
        )

    with pytest.raises(AgentSpecSerializationError):
        execute(connection=object(), workflow=workflow, inputs={})


def test_loop_rejects_nonfinite_embedded_body_spec(database_llm) -> None:
    contract = {"done": "string", "value": {"type": "number", "default": float("nan")}}
    body = agent(
        name="step",
        llm=database_llm,
        instruction="Update {{done}} and {{value}}",
        inputs=contract,
        outputs=contract,
    )

    with pytest.raises(PatternCompilationError):
        loop(name="repeat", body=body, state_key="done", continue_value="N", max_iterations=2)


@pytest.mark.parametrize(
    "payload",
    [
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e400}',
        '{"value": -1e400}',
        '{"synthetic-sensitive-value": invalid}',
    ],
)
def test_invalid_serializer_json_raises_redacted_error(make_agent, monkeypatch, payload) -> None:
    class FakeSerializer:
        def to_json(self, component):
            return payload

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FakeSerializer)

    with pytest.raises(AgentSpecSerializationError) as caught:
        _serialize_workflow(make_agent("agent"))

    assert caught.value.__cause__ is None
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))


def test_json_validation_preserves_serializer_text_and_allows_finite_numbers(
    make_agent, monkeypatch
) -> None:
    payload = '{ "value": [0, -0.0, 1.25, 1e308], "labels": ["NaN", "Infinity", "-Infinity"] }'

    class FakeSerializer:
        def to_json(self, component):
            return payload

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", FakeSerializer)

    assert _serialize_workflow(make_agent("agent")) == payload


def _components(value):
    if isinstance(value, dict):
        if "component_type" in value:
            yield value
        for child in value.values():
            yield from _components(child)
    elif isinstance(value, list):
        for child in value:
            yield from _components(child)


@pytest.mark.parametrize("kind", ["llm", "agent"])
@pytest.mark.parametrize("shape", ["parallel", "nested", "nested_local"])
def test_parallel_shared_model_is_relocated_without_changing_identity(database_llm, kind, shape):
    def step(name, model):
        kwargs = dict(name=name, llm=model, inputs={"text": "string"}, outputs={name: "string"})
        if kind == "agent":
            return agent(instruction="Copy {{text}}.", **kwargs)
        return llm_node(prompt="Copy {{text}}.", **kwargs)

    workflow = parallel(
        name="inner", steps=[step("first", database_llm), step("second", database_llm)]
    )
    if shape != "parallel":
        other_model = (
            llm_config(
                provider="openai",
                model_id="other",
                url="https://example.invalid/chat",
                credential_name="TEST_CREDENTIAL",
            )
            if shape == "nested_local"
            else database_llm
        )
        workflow = parallel(name="outer", steps=[workflow, step("third", other_model)])

    component = _unwrap_executable(workflow)
    original = AgentSpecSerializer().to_json(component)
    before = json.loads(original)
    assert any(
        entry.get("component_type") == "DbmsVectorChainLlmConfig"
        for value in _components(before)
        if value["component_type"] == "ParallelFlowNode"
        for entry in value.get("$referenced_components", {}).values()
    )

    serialized = _serialize_workflow(workflow)
    after = json.loads(serialized)
    assert AgentSpecSerializer().to_json(component) == original
    assert AgentSpecDeserializer().from_json(serialized) == AgentSpecDeserializer().from_json(
        original
    )
    for value in _components(after):
        if value["component_type"] == "ParallelFlowNode":
            assert all(
                entry.get("component_type") != "DbmsVectorChainLlmConfig"
                for entry in value.get("$referenced_components", {}).values()
            )
    configurations_before = [
        value
        for value in _components(before)
        if value["component_type"] == "DbmsVectorChainLlmConfig"
    ]
    configurations_after = [
        value
        for value in _components(after)
        if value["component_type"] == "DbmsVectorChainLlmConfig"
    ]
    assert sorted(configurations_after, key=lambda value: value["id"]) == sorted(
        configurations_before, key=lambda value: value["id"]
    )
    assert len(configurations_after) == (2 if shape == "nested_local" else 1)


def _shared_model_document():
    return {
        "component_type": "Flow",
        "id": "flow",
        "nodes": [
            {
                "component_type": "ParallelFlowNode",
                "id": "parallel",
                "$referenced_components": {
                    "model": {"component_type": "DbmsVectorChainLlmConfig", "id": "model"},
                    "child": {"component_type": "Flow", "id": "child"},
                },
                "subflows": [{"$component_ref": "child"}],
            }
        ],
    }


def _serialize_document(document, make_agent, monkeypatch):
    class Serializer:
        def to_json(self, component):
            return json.dumps(document)

    monkeypatch.setattr(serialization_module, "AgentSpecSerializer", Serializer)
    return _serialize_workflow(make_agent("test"))


def test_shared_model_relocation_preserves_application_data_and_other_definitions(
    make_agent,
    monkeypatch,
):
    document = _shared_model_document()
    application_value = _shared_model_document()
    document.update(
        inputs=[{"title": "input", "default": application_value}],
        outputs=[{"title": "output", "default": application_value}],
        metadata={"sample": application_value},
        configuration={"sample": application_value},
    )
    model = document["nodes"][0]["$referenced_components"]["model"]
    model["default_generation_parameters"] = {"sample": application_value}
    before = deepcopy(document)
    after = json.loads(_serialize_document(document, make_agent, monkeypatch))

    assert document == before
    assert after["$referenced_components"] == {"model": model}
    assert after["nodes"][0]["$referenced_components"] == {
        "child": {"component_type": "Flow", "id": "child"}
    }
    for key in ("inputs", "outputs", "metadata", "configuration"):
        assert after[key] == before[key]
    assert after["nodes"][0]["subflows"] == before["nodes"][0]["subflows"]


@pytest.mark.parametrize("same_definition", [True, False])
def test_shared_model_relocation_rejects_reference_id_collisions(
    make_agent,
    monkeypatch,
    same_definition,
):
    document = _shared_model_document()
    definition = deepcopy(document["nodes"][0]["$referenced_components"]["model"])
    if not same_definition:
        definition["model_id"] = "synthetic-sensitive-value"
    document["$referenced_components"] = {"model": definition}

    with pytest.raises(AgentSpecSerializationError) as caught:
        _serialize_document(document, make_agent, monkeypatch)
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))


def test_parallel_shared_model_requires_an_enclosing_flow(make_agent, monkeypatch):
    document = _shared_model_document()["nodes"][0]
    with pytest.raises(AgentSpecSerializationError):
        _serialize_document(document, make_agent, monkeypatch)
