##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for model-driven coordination and stored-agent capabilities."""

import json
import traceback

import pytest
from pyagentspec import AgentSpecDeserializer, AgentSpecSerializer
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import FlowNode

from oracle_db_agent_sdk import (
    BuildingBlockError,
    PatternCompilationError,
    agent,
    agent_tool,
    orchestrator,
    plsql_tool,
    sequential,
)
from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import _unwrap_executable, _unwrap_tool


def capability():
    return plsql_tool(
        name="lookup",
        schema="APP",
        procedure="LOOKUP",
        inputs={"case_id": "string"},
        outputs={"status": "string"},
    )


def build_coordinator(database_llm, **changes):
    arguments = dict(
        name="coordinator",
        llm=database_llm,
        instruction="Resolve {{value}} using the tools.",
        inputs={"value": "string"},
        outputs={"value": "string"},
        tools=[capability()],
        description="Coordinate support work.",
    )
    arguments.update(changes)
    return orchestrator(**arguments)


def builtin_definitions(value):
    if isinstance(value, dict):
        if value.get("component_type") == "BuiltinTool":
            yield value
        for child in value.values():
            yield from builtin_definitions(child)
    elif isinstance(value, list):
        for child in value:
            yield from builtin_definitions(child)


def test_orchestrator_reuses_manager_with_declared_tools(database_llm) -> None:
    first = capability()
    second = agent_tool(
        name="assess",
        schema="APP",
        agent_name="ASSESS_CASE",
        inputs={"case_id": "string"},
        outputs={"assessment": "string"},
    )
    workflow = build_coordinator(database_llm, tools=[first, second])
    component = _unwrap_executable(workflow)
    assert isinstance(component, Flow)
    assert component.description == "Coordinate support work."
    decoded = AgentSpecDeserializer().from_json(_serialize_workflow(workflow))
    assert isinstance(decoded, Flow)
    tools = list(builtin_definitions(json.loads(_serialize_workflow(workflow))))
    assert sorted(tool["tool_type"] for tool in tools) == ["agent_exec", "plsql_exec"]
    assert len([n for n in decoded.nodes if n.name == "orchestrator-plan"]) == 1


@pytest.mark.parametrize("placement", ["only", "first", "last"])
def test_agent_rejects_agent_backed_tools(database_llm, placement) -> None:
    specialist = agent_tool(
        name="assess",
        schema="APP",
        agent_name="ASSESS_CASE",
        inputs={"case_id": "string"},
        outputs={"assessment": "string"},
    )
    tools = [specialist]
    if placement == "first":
        tools.append(capability())
    elif placement == "last":
        tools.insert(0, capability())

    with pytest.raises(BuildingBlockError, match="use orchestrator"):
        agent(
            name="support",
            llm=database_llm,
            instruction="Resolve case {{case_id}}.",
            inputs={"case_id": "string"},
            outputs={"resolution": "string"},
            tools=tools,
        )


def test_orchestrator_accepts_agent_backed_tool_without_procedure(database_llm) -> None:
    specialist = agent_tool(
        name="assess",
        schema="APP",
        agent_name="ASSESS_CASE",
        inputs={"case_id": "string"},
        outputs={"assessment": "string"},
    )
    workflow = build_coordinator(database_llm, tools=[specialist])

    decoded = AgentSpecDeserializer().from_json(_serialize_workflow(workflow))
    assert isinstance(decoded, Flow)
    tools = list(builtin_definitions(json.loads(_serialize_workflow(workflow))))
    assert [tool["tool_type"] for tool in tools] == ["agent_exec"]
    assert tools[0]["configuration"] == {"schema_name": "APP", "agent_name": "ASSESS_CASE"}


def test_orchestrator_nests_as_flow_node(database_llm, make_agent) -> None:
    coordinator = build_coordinator(database_llm)
    workflow = sequential(name="review", steps=[make_agent("first"), coordinator])
    assert isinstance(_unwrap_executable(workflow).nodes[2], FlowNode)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"name": " "},
        {"name": None},
        {"instruction": ""},
        {"instruction": None},
        {"inputs": {}},
        {"inputs": []},
        {"outputs": {}},
        {"outputs": []},
        {"tools": []},
        {"tools": "tool"},
        {"tools": None},
        {"tools": [object()]},
        {"llm": object()},
    ],
)
def test_orchestrator_rejects_invalid_configuration(database_llm, changes) -> None:
    with pytest.raises(PatternCompilationError):
        build_coordinator(database_llm, **changes)


def test_orchestrator_rejects_executables_as_tools(database_llm, make_agent) -> None:
    with pytest.raises(PatternCompilationError):
        build_coordinator(database_llm, tools=[make_agent("worker")])


def test_orchestrator_rejects_raw_pyagentspec_tools(database_llm) -> None:
    with pytest.raises(PatternCompilationError):
        build_coordinator(database_llm, tools=[_unwrap_tool(capability())])


def test_orchestrator_errors_do_not_leak_validation_values(database_llm) -> None:
    private = "synthetic-sensitive-value"
    with pytest.raises(PatternCompilationError) as caught:
        build_coordinator(database_llm, inputs={"": {"type": "string", "default": private}})
    assert private not in "".join(traceback.format_exception(caught.value))


def test_agent_tool_references_stored_identity_without_copying_graph() -> None:
    tool = agent_tool(
        name="assess",
        schema="APP",
        agent_name="ASSESS_CASE",
        inputs={"case_id": "string"},
        outputs={"assessment": "string"},
        description="Assess a support case.",
    )
    component = _unwrap_tool(tool)
    assert component.configuration == {"schema_name": "APP", "agent_name": "ASSESS_CASE"}
    assert component.tool_type == "agent_exec"
    assert component.requires_confirmation is False
    assert component.description == "Assess a support case."
    assert [(prop.title, prop.type) for prop in component.inputs] == [("case_id", "string")]
    assert "subflow" not in json.loads(AgentSpecSerializer().to_json(component))


@pytest.mark.parametrize("changes", [{"name": ""}, {"schema": " "}, {"agent_name": None}])
def test_agent_tool_rejects_missing_identity(changes) -> None:
    arguments = dict(name="assess", schema="APP", agent_name="ASSESS", inputs={}, outputs={})
    arguments.update(changes)
    with pytest.raises(BuildingBlockError, match="reference"):
        agent_tool(**arguments)


def test_agent_tool_redacts_contract_errors() -> None:
    private = "synthetic-sensitive-value"
    with pytest.raises(BuildingBlockError) as caught:
        agent_tool(
            name="assess",
            schema="APP",
            agent_name="ASSESS",
            inputs={"": {"type": "string", "default": private}},
            outputs={},
        )
    assert private not in "".join(traceback.format_exception(caught.value))


def test_agent_tool_wraps_builtin_validation_failure(monkeypatch) -> None:
    import oracle_db_agent_sdk.building_blocks as blocks

    def fail(**kwargs):
        raise ValueError("synthetic-sensitive-value")

    monkeypatch.setattr(blocks, "BuiltinTool", fail)
    with pytest.raises(BuildingBlockError, match="stored agent tool") as caught:
        agent_tool(name="a", schema="APP", agent_name="A", inputs={}, outputs={})
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))
