##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Execute serialized control/data paths with deterministic model/agent doubles.

These tests prove graph selection and wiring, not Oracle or provider behavior.
The runner interprets only standard AgentSpec nodes used by these test graphs.
"""

import json
from copy import deepcopy

import pytest
from pyagentspec import AgentSpecDeserializer, AgentSpecSerializer
from pyagentspec.flows.nodes import AgentNode, BranchingNode, EndNode, FlowNode, LlmNode, StartNode
from pyagentspec.property import Property

import oracle_db_agent_sdk.executor as execution
from oracle_db_agent_sdk import (
    BuildingBlockError,
    LoopExecutionError,
    OrchestrationExecutionError,
    PatternCompilationError,
    agent,
    agent_tool,
    execute,
    loop,
    orchestrator,
    sequential,
)
from oracle_db_agent_sdk._flow_status import public_outputs
from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import _unwrap_executable


class GraphRunner:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []
        self.agent_results = {}
        self.plans = []
        self.finals = []

    def run(self, flow, inputs):
        return self._run(flow, inputs)[0]

    def _run(self, flow, inputs):
        values = {flow.start_node.id: inputs}
        current = flow.start_node
        for _ in range(1000):
            supplied = {}
            for prop in current.inputs or ():
                if prop.default is not Property.empty_default:
                    supplied[prop.title] = deepcopy(prop.default)
            for edge in flow.data_flow_connections:
                if edge.destination_node.id == current.id and edge.source_node.id in values:
                    supplied[edge.destination_input] = values[edge.source_node.id][
                        edge.source_output
                    ]
            branch = "next"
            if isinstance(current, StartNode):
                result = inputs
            elif isinstance(current, EndNode):
                return supplied, current.branch_name
            elif isinstance(current, FlowNode):
                result, branch = self._run(current.subflow, supplied)
            elif isinstance(current, AgentNode) and not current.name.startswith("orchestrator-"):
                self.calls.append((current.agent.name, supplied))
                result = self.agent_results.get(
                    current.agent.name, {"value": supplied["value"] + ":" + current.agent.name}
                )
            elif isinstance(current, BranchingNode):
                selector = supplied[current.inputs[0].title]
                branch = current.mapping.get(selector, "default")
                result = {}
            elif isinstance(current, (LlmNode, AgentNode)):
                memory_key = next(k for k in supplied if k.endswith("memory"))
                history = json.loads(supplied[memory_key] or "[]")
                if current.name == "orchestrator-plan":
                    self.plans.append(supplied)
                    previous = next(v for k, v in supplied.items() if k.endswith("previous_action"))
                    if previous:
                        called = next(
                            k for k, v in supplied.items() if "_called_" in k and v == previous
                        )
                        index = called.rsplit("_", 1)[-1]
                        value = next(v for k, v in supplied.items() if f"_result_{index}_" in k)
                        history.append({"capability": previous, "value": value})
                    action = next(self.actions)
                    result = {
                        p.title: action if p.title.endswith("action") else json.dumps(history)
                        for p in current.outputs
                    }
                elif current.name.startswith("orchestrator-arguments-"):
                    value = history[-1]["value"] if history else supplied["value"]
                    result = {p.title: value for p in current.outputs}
                else:
                    self.finals.append(supplied)
                    result = {"value": history[-1]["value"] if history else supplied["value"]}
            else:
                raise AssertionError(type(current))
            values[current.id] = result
            edges = [
                e
                for e in flow.control_flow_connections
                if e.from_node.id == current.id and (e.from_branch or "next") == branch
            ]
            assert len(edges) == 1, (current.name, branch)
            current = edges[0].to_node
        raise AssertionError("graph did not terminate")


def build(database_llm, *, tools=None):
    if tools is None:
        child = agent(
            name="specialist",
            llm=database_llm,
            instruction="Handle {{value}}.",
            inputs={"value": "string"},
            outputs={"value": "string"},
        )
        tools = [agent_tool(name="worker", agent=child)]
    return orchestrator(
        name="coordinator",
        llm=database_llm,
        instruction="Handle {{value}}.",
        inputs={"value": "string"},
        outputs={"value": "string"},
        tools=tools,
    )


def install_runner(monkeypatch, actions):
    runner = GraphRunner(actions)
    submissions = []

    def submit(connection, *, agent_spec_json, inputs_json):
        submissions.append(agent_spec_json)
        graph = AgentSpecDeserializer().from_json(agent_spec_json)
        inputs = {item["name"]: item["value"] for item in json.loads(inputs_json)}
        return json.dumps(runner.run(graph, inputs))

    monkeypatch.setattr(execution, "execute_immediate", submit)
    return runner, submissions


def test_repeated_delegation_uses_real_prior_results_and_submits_once(database_llm, monkeypatch):
    runner, submissions = install_runner(monkeypatch, ["worker", "worker", "finish"])
    result = execute(connection=object(), workflow=build(database_llm), inputs={"value": "start"})
    assert result == {"value": "start:specialist:specialist"}
    assert runner.calls == [
        ("specialist", {"value": "start"}),
        ("specialist", {"value": "start:specialist"}),
    ]
    assert len(submissions) == 1
    assert len(runner.plans) == 3 and len(runner.finals) == 1
    memory = next(v for k, v in runner.finals[0].items() if k.endswith("memory"))
    assert [item["value"] for item in json.loads(memory)] == [
        "start:specialist",
        "start:specialist:specialist",
    ]


def test_finish_without_delegation(database_llm, monkeypatch):
    runner, _ = install_runner(monkeypatch, ["finish"])
    assert execute(connection=object(), workflow=build(database_llm), inputs={"value": "x"}) == {
        "value": "x"
    }
    assert runner.calls == []


def test_only_selected_capability_runs(database_llm, monkeypatch, make_agent):
    runner, _ = install_runner(monkeypatch, ["second", "finish"])
    tools = [
        agent_tool(name="first", agent=make_agent("FIRST")),
        agent_tool(name="second", agent=make_agent("SECOND")),
    ]
    result = execute(
        connection=object(), workflow=build(database_llm, tools=tools), inputs={"value": "x"}
    )
    assert result == {"value": "x:SECOND"}
    assert runner.calls == [("SECOND", {"value": "x"})]


def test_failure_in_embedded_orchestrator_propagates(database_llm, monkeypatch):
    inner = build(database_llm)
    outer = build(database_llm, tools=[agent_tool(name="worker", agent=inner)])
    runner, _ = install_runner(monkeypatch, ["worker", "unknown"])
    with pytest.raises(OrchestrationExecutionError, match="unknown capability"):
        execute(connection=object(), workflow=outer, inputs={"value": "x"})
    assert runner.calls == []
    assert runner.finals == []


@pytest.mark.parametrize("child_kind", ["loop", "orchestrator"])
def test_embedded_workflow_branches_have_exactly_one_continuation(
    database_llm, make_agent, child_kind
):
    child = (
        loop(
            name="repeat",
            body=make_agent("specialist"),
            state_key="value",
            continue_value="continue",
            max_iterations=1,
        )
        if child_kind == "loop"
        else build(database_llm)
    )
    workflow = build(database_llm, tools=[agent_tool(name="worker", agent=child)])
    graph = AgentSpecDeserializer().from_json(_serialize_workflow(workflow))

    def check(flow):
        for node in flow.nodes:
            if isinstance(node, FlowNode):
                wired = [
                    edge.from_branch or "next"
                    for edge in flow.control_flow_connections
                    if edge.from_node.id == node.id
                ]
                assert sorted(wired) == sorted(node.branches), node.name
                check(node.subflow)

    calls = [node for node in graph.nodes if isinstance(node, FlowNode)]
    assert len(calls) == 1
    assert all(any(branch != "next" for branch in node.branches) for node in calls)
    check(graph)


def test_embedded_loop_failure_stops_parent(database_llm, monkeypatch, make_agent):
    child = loop(
        name="repeat",
        body=make_agent("specialist"),
        state_key="value",
        continue_value="start",
        max_iterations=1,
    )
    workflow = build(database_llm, tools=[agent_tool(name="worker", agent=child)])
    runner, submissions = install_runner(monkeypatch, ["worker"])
    runner.agent_results["SPECIALIST"] = {"value": "start"}
    with pytest.raises(LoopExecutionError, match="iteration limit"):
        execute(connection=object(), workflow=workflow, inputs={"value": "start"})
    assert runner.calls == [("SPECIALIST", {"value": "start"})]
    assert runner.finals == []
    assert len(submissions) == 1


@pytest.mark.parametrize("count", [12, 25])
def test_delegation_continues_beyond_previous_bound(database_llm, monkeypatch, count):
    runner, submissions = install_runner(monkeypatch, ["worker"] * count + ["finish"])
    result = execute(connection=object(), workflow=build(database_llm), inputs={"value": "start"})
    assert result == {"value": "start" + ":specialist" * count}
    assert len(runner.calls) == count
    assert len(runner.plans) == count + 1
    assert len(submissions) == 1


def test_invalid_choice_after_many_calls_stops_before_another_call(database_llm, monkeypatch):
    runner, _ = install_runner(monkeypatch, ["worker"] * 12 + ["unknown"])
    with pytest.raises(OrchestrationExecutionError, match="unknown capability"):
        execute(connection=object(), workflow=build(database_llm), inputs={"value": "start"})
    assert len(runner.calls) == 12
    assert runner.finals == []


def test_nested_failure_stops_following_application_step(database_llm, monkeypatch, make_agent):
    runner, _ = install_runner(monkeypatch, ["unknown"])
    workflow = sequential(name="outer", steps=[build(database_llm), make_agent("after")])
    with pytest.raises(OrchestrationExecutionError, match="unknown capability"):
        execute(connection=object(), workflow=workflow, inputs={"value": "x"})
    assert runner.calls == []


def test_embedded_definitions_are_shared_without_mutating_source(database_llm, make_agent):
    child = make_agent("specialist")
    before = AgentSpecSerializer().to_json(_unwrap_executable(child))
    workflow = build(database_llm, tools=[agent_tool(name="worker", agent=child)])
    graph = AgentSpecDeserializer().from_json(_serialize_workflow(workflow))
    calls = [node for node in graph.nodes if isinstance(node, FlowNode)]
    assert len(calls) == 1
    assert any(edge.from_node.id == calls[0].id for edge in graph.control_flow_connections)
    assert AgentSpecSerializer().to_json(_unwrap_executable(child)) == before
    assert [p.title for p in public_outputs(graph)] == ["value"]
    spec = json.loads(_serialize_workflow(workflow))
    assert "agent_exec" not in json.dumps(spec)
    assert "transforms" not in json.dumps(spec)


def test_orchestrator_rejects_public_iteration_limit(database_llm, make_agent):
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_iterations'"):
        orchestrator(
            name="coordinator",
            llm=database_llm,
            instruction="Handle {{value}}.",
            inputs={"value": "string"},
            outputs={"value": "string"},
            tools=[agent_tool(name="worker", agent=make_agent("specialist"))],
            max_iterations=1,
        )


def test_invalid_embedded_references_and_ambiguous_forms(database_llm, make_agent):
    with pytest.raises(BuildingBlockError):
        agent_tool(name="bad", agent=object())
    with pytest.raises(BuildingBlockError, match="description"):
        agent_tool(name="bad", agent=make_agent("child"), description=object())
    with pytest.raises(BuildingBlockError, match="Choose"):
        agent_tool(name="bad", agent=make_agent("child"), schema="APP")
    for name in ("worker", "finish"):
        tool = agent_tool(name=name, agent=make_agent("child"))
        with pytest.raises(PatternCompilationError, match="unique"):
            build(database_llm, tools=[tool, tool] if name == "worker" else [tool])


def test_capability_without_arguments_and_internal_name_collision(database_llm):
    child = agent(
        name="constant",
        llm=database_llm,
        instruction="Return a greeting.",
        inputs={},
        outputs={"value": "string"},
    )
    workflow = orchestrator(
        name="zero_args",
        llm=database_llm,
        instruction="Handle {{__sdk_orch_action}}.",
        inputs={"__sdk_orch_action": "string"},
        outputs={"value": "string"},
        tools=[agent_tool(name="constant", agent=child)],
    )
    graph = AgentSpecDeserializer().from_json(_serialize_workflow(workflow))
    assert not any(n.name.startswith("orchestrator-arguments-") for n in graph.nodes)
    planner = next(n for n in graph.nodes if n.name == "orchestrator-plan")
    assert planner.outputs[0].title != "__sdk_orch_action"


def test_rich_contracts_survive_aliasing_and_feedback(database_llm):
    schema = {"type": "string", "anyOf": [{"type": "string", "maxLength": 10}], "default": "ok"}
    child = agent(
        name="child",
        llm=database_llm,
        instruction="Handle {{value}}.",
        inputs={"value": schema},
        outputs={"value": schema},
    )
    before = AgentSpecSerializer().to_json(_unwrap_executable(child))
    workflow = orchestrator(
        name="rich",
        llm=database_llm,
        instruction="Handle {{value}}.",
        inputs={"value": schema},
        outputs={"value": schema},
        tools=[agent_tool(name="child", agent=child)],
    )
    wire = json.loads(_serialize_workflow(workflow))
    graph = AgentSpecDeserializer().from_json(json.dumps(wire))
    assert public_outputs(graph)[0].json_schema == {**schema, "title": "value"}
    # Dependency validators normalize rich schemas during deserialization;
    # the contract sent to the database must retain the original constraints.
    assert wire["inputs"][0] == {**schema, "title": "value"}
    assert AgentSpecSerializer().to_json(_unwrap_executable(child)) == before
