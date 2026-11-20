##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for bounded-loop compilation and ordinary pattern composition."""

import json
import traceback
from copy import deepcopy

import pytest
from pyagentspec import AgentSpecDeserializer, AgentSpecSerializer
from pyagentspec.component import Component
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import BranchingNode, EndNode, FlowNode, ParallelFlowNode

import oracle_db_agent_sdk._loop_compiler as loop_compiler
from oracle_db_agent_sdk import (
    IncompatibleLoopContractError,
    PatternCompilationError,
    agent,
    llm_node,
    loop,
    parallel,
    routing,
    sequential,
)
from oracle_db_agent_sdk._flow_status import public_outputs, status_field
from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import (
    _unwrap_executable,
    _wrap_executable,
)


@pytest.fixture
def loop_body(database_llm):
    contract = {"counter": "integer", "limit": "integer", "done": "string", "trace": "string"}
    return agent(
        name="count",
        llm=database_llm,
        instruction="Increment {{counter}} toward {{limit}} and update {{done}} and {{trace}}.",
        inputs=contract,
        outputs=contract,
    )


def build_loop(component, **changes):
    arguments = dict(
        name="count_loop", body=component, state_key="done", continue_value="N", max_iterations=10
    )
    arguments.update(changes)
    return loop(**arguments)


def _bodies(flow):
    return [node for node in flow.nodes if node.name.startswith("loop-body-")]


def _conditions(flow):
    return [node for node in flow.nodes if node.name.startswith("loop-condition-")]


def _assert_acyclic(flow):
    """Independently topologically sort each emitted control graph."""
    incoming = {node.id: 0 for node in flow.nodes}
    successors = {node.id: [] for node in flow.nodes}
    for edge in flow.control_flow_connections:
        successors[edge.from_node.id].append(edge.to_node.id)
        incoming[edge.to_node.id] += 1
    ready = [node_id for node_id, count in incoming.items() if not count]
    visited = set()
    while ready:
        source = ready.pop()
        visited.add(source)
        for destination in successors[source]:
            incoming[destination] -= 1
            if incoming[destination] == 0:
                ready.append(destination)
    assert visited == set(incoming), "control graph has a cycle"
    for node in flow.nodes:
        if isinstance(node, FlowNode):
            _assert_acyclic(node.subflow)
        elif isinstance(node, ParallelFlowNode):
            for subflow in node.subflows:
                _assert_acyclic(subflow)


def _component_ids(component):
    ids = set()
    seen = set()

    def visit(value):
        if isinstance(value, Component):
            if id(value) in seen:
                return
            seen.add(id(value))
            ids.add(value.id)
            for field in type(value).model_fields:
                visit(getattr(value, field))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(component)
    return ids


def test_loop_emits_bounded_forward_graph_and_preserves_body(loop_body) -> None:
    original = _serialize_workflow(loop_body)
    flow = _unwrap_executable(build_loop(loop_body, max_iterations=3))
    assert isinstance(flow, Flow)
    fields = ["counter", "limit", "done", "trace"]
    assert [p.title for p in flow.inputs] == fields
    assert [p.title for p in public_outputs(flow)] == fields
    assert [p.title for p in flow.outputs] == [*fields, status_field(flow)]
    bodies, gates = _bodies(flow), _conditions(flow)
    assert len(bodies) == 3 and len(gates) == 4
    assert all(gate.mapping == {"N": "loop_continue"} for gate in gates)
    _assert_acyclic(flow)
    edges = {(e.from_node.id, e.from_branch): e.to_node for e in flow.control_flow_connections}
    success = next(node for node in flow.nodes if node.name == "loop-end")
    exhausted = edges[gates[-1].id, "loop_continue"]
    assert isinstance(exhausted, EndNode) and exhausted is not success
    assert exhausted.outputs[0].json_schema["default"] == "exhausted"
    assert edges[flow.start_node.id, None] is gates[0]
    for index, gate in enumerate(gates):
        assert edges[gate.id, "default"] is success
        if index < len(bodies):
            assert edges[gate.id, "loop_continue"] is bodies[index]
            assert edges[bodies[index].id, "next"] is gates[index + 1]
    data = {
        (e.source_node.id, e.source_output, e.destination_node.id, e.destination_input)
        for e in flow.data_flow_connections
    }
    expected = set()
    for index, source in enumerate([flow.start_node, *bodies]):
        expected.add((source.id, "done", gates[index].id, "done"))
        expected.update((source.id, field, success.id, field) for field in fields)
        if index < len(bodies):
            expected.update((source.id, field, bodies[index].id, field) for field in fields)
    assert data == expected
    occurrences = [_component_ids(body) for body in bodies]
    original_ids = _component_ids(_unwrap_executable(loop_body))
    for index, ids in enumerate(occurrences):
        assert ids.isdisjoint(original_ids)
        assert all(bodies[index].id not in other for other in occurrences[index + 1 :])
    assert all(body.agent is bodies[0].agent for body in bodies)
    assert _serialize_workflow(loop_body) == original


@pytest.mark.parametrize("body_kind", ["agent", "flow", "node", "nested_loop"])
def test_loop_shares_definitions_but_preserves_call_sites(loop_body, database_llm, body_kind):
    body = loop_body
    if body_kind == "flow":
        body = sequential(name="body", steps=[loop_body, loop_body])
    elif body_kind == "node":
        body = llm_node(
            name="review",
            llm=database_llm,
            prompt="Update {{done}}.",
            inputs={"done": "string"},
            outputs={"done": "string"},
        )
    elif body_kind == "nested_loop":
        body = build_loop(loop_body, max_iterations=2)
    original = AgentSpecSerializer().to_json(_unwrap_executable(body))
    workflow = build_loop(body, max_iterations=3)
    graph = _unwrap_executable(workflow)
    spec = _serialize_workflow(workflow)
    decoded = AgentSpecDeserializer().from_json(spec)
    field = "agent" if body_kind == "agent" else "llm_config" if body_kind == "node" else "subflow"
    for plan in (graph, decoded):
        calls = _bodies(plan)
        assert len(calls) == len({node.id for node in calls}) == 3
        assert [node.name for node in calls] == [f"loop-body-{i}" for i in range(3)]
        assert len({getattr(node, field).id for node in calls}) == 1
        _assert_acyclic(plan)
        if body_kind == "nested_loop":
            assert all(status_field(node) for node in calls)
            checks = [node for node in plan.nodes if node.name.startswith("loop-status-check-")]
            assert len(checks) == 3

    shared_id = getattr(_bodies(graph)[0], field).id
    definitions = []

    def collect(value):
        if isinstance(value, dict):
            if value.get("id") == shared_id and "component_type" in value:
                definitions.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(json.loads(spec))
    assert len(definitions) == 1
    assert shared_id in json.loads(spec)["$referenced_components"]
    other = _unwrap_executable(build_loop(body, max_iterations=3))
    assert _component_ids(graph).isdisjoint(_component_ids(other))
    assert AgentSpecSerializer().to_json(_unwrap_executable(body)) == original


def test_loop_round_trips_without_native_loop_metadata(loop_body) -> None:
    workflow = build_loop(loop_body, max_iterations=3)
    spec = _serialize_workflow(workflow)
    decoded = AgentSpecDeserializer().from_json(spec)
    assert isinstance(decoded, Flow)
    _assert_acyclic(decoded)
    assert len(_bodies(decoded)) == 3 and len(_conditions(decoded)) == 4
    assert status_field(decoded) == status_field(_unwrap_executable(workflow))
    assert "agent_sql_pattern_loop" not in spec
    assert "loop_gate_id" not in spec
    assert "loop_max_iterations" not in spec


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"name": " "},
        {"name": 1},
        {"body": object()},
        {"state_key": ""},
        {"state_key": None},
        {"continue_value": ""},
        {"continue_value": None},
        {"continue_value": 1},
        {"max_iterations": 0},
        {"max_iterations": -1},
        {"max_iterations": 1.5},
        {"max_iterations": True},
        {"max_iterations": "10"},
        {"max_iterations": 257},
    ],
)
def test_loop_rejects_invalid_configuration(loop_body, changes) -> None:
    with pytest.raises(PatternCompilationError):
        build_loop(loop_body, **changes)


@pytest.mark.parametrize(
    ("inputs", "outputs", "state_key"),
    [
        ({}, {}, "done"),
        ({"done": "string"}, {"other": "string"}, "done"),
        ({"done": "string"}, {"done": "integer"}, "done"),
        ({"done": "integer"}, {"done": "integer"}, "done"),
        ({"done": "string"}, {"done": "string"}, "missing"),
    ],
)
def test_loop_rejects_incompatible_feedback(database_llm, inputs, outputs, state_key) -> None:
    body = agent(
        name="step",
        llm=database_llm,
        instruction=" ".join("{{" + title + "}}" for title in inputs) or "Do work.",
        inputs=inputs,
        outputs=outputs,
    )
    with pytest.raises(IncompatibleLoopContractError):
        build_loop(body, state_key=state_key)


def test_loop_rejects_duplicate_fields(loop_body) -> None:
    component = _unwrap_executable(loop_body)
    component.inputs.append(component.inputs[0])
    with pytest.raises(IncompatibleLoopContractError, match="unique"):
        build_loop(loop_body)


def test_loop_requires_explicit_field_types(loop_body) -> None:
    component = _unwrap_executable(loop_body)
    component.inputs[0].type = None
    component.outputs[0].type = None
    with pytest.raises(IncompatibleLoopContractError, match="explicit JSON types"):
        build_loop(loop_body)


def test_loop_supports_nullable_non_control_fields(database_llm) -> None:
    contract = {"done": "string", "result": {"type": ["string", "null"]}}
    body = agent(
        name="step",
        llm=database_llm,
        instruction="Update {{done}} and {{result}}.",
        inputs=contract,
        outputs=contract,
    )
    plan = _unwrap_executable(build_loop(body))
    assert plan.inputs[1].type == ["string", "null"]


def test_loop_rejects_raw_pyagentspec_body(loop_body) -> None:
    with pytest.raises(PatternCompilationError, match="building block"):
        build_loop(_unwrap_executable(loop_body))


def test_loop_accepts_nested_sequence_as_body(loop_body) -> None:
    body = sequential(name="body", steps=[loop_body, loop_body])
    workflow = build_loop(body)
    plan = _unwrap_executable(workflow)
    serialized_body = _serialize_workflow(_wrap_executable(_bodies(plan)[0].subflow))
    assert isinstance(AgentSpecDeserializer().from_json(serialized_body), Flow)


def test_loop_can_be_nested_in_native_sequence(loop_body) -> None:
    first, second = build_loop(loop_body, name="first"), build_loop(loop_body, name="second")
    workflow = sequential(name="both", steps=[first, second])
    plan = _unwrap_executable(workflow)
    assert isinstance(plan, Flow)
    assert len([node for node in plan.nodes if isinstance(node, FlowNode)]) == 2
    assert status_field(plan)
    _assert_acyclic(plan)


def test_nested_parallel_uses_native_flow(loop_body, make_agent) -> None:
    repeating = build_loop(loop_body)
    independent = make_agent("independent", inputs=(), outputs=(("other", "string"),))
    workflow = parallel(name="parallel_test", steps=[repeating, independent])
    assert isinstance(_unwrap_executable(workflow), Flow)


def test_nested_routing_uses_native_flow(loop_body, database_llm) -> None:
    repeating = build_loop(loop_body)
    selector = agent(
        name="selector",
        llm=database_llm,
        instruction="Choose a route for {{done}}, {{counter}}, {{limit}}, {{trace}}.",
        inputs={p.title: p.json_schema for p in _unwrap_executable(repeating).inputs},
        outputs={"route": "string"},
    )
    workflow = routing(
        selector,
        name="routing_test",
        route_key="route",
        routes={"repeat": repeating},
        default=repeating,
    )
    assert isinstance(_unwrap_executable(workflow), Flow)


def test_loop_compilation_errors_are_redacted(loop_body, monkeypatch) -> None:
    def fail(*args):
        raise ValueError("synthetic-sensitive-value")

    monkeypatch.setattr(loop_compiler, "_serialize_workflow", fail)
    with pytest.raises(PatternCompilationError) as caught:
        build_loop(loop_body)
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("keyword", ["allOf", "oneOf"])
def test_unsupported_loop_contract_raises_public_compilation_error(database_llm, keyword) -> None:
    contract = {
        "done": "string",
        "value": {"type": "string", keyword: [{"type": "string"}]},
    }
    body = agent(
        name="step",
        llm=database_llm,
        instruction="Update {{done}} and {{value}}.",
        inputs=contract,
        outputs=contract,
    )

    with pytest.raises(PatternCompilationError) as caught:
        build_loop(body)

    assert caught.value.__cause__ is None
    assert "NotImplementedError" not in "".join(traceback.format_exception(caught.value))


def test_loop_contract_validation_diagnostics_are_redacted(loop_body, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise ValueError("synthetic-sensitive-schema-default")

    monkeypatch.setattr(loop_compiler, "_loop_state_contract", fail)

    with pytest.raises(PatternCompilationError) as caught:
        build_loop(loop_body)

    assert caught.value.__cause__ is None
    assert "synthetic-sensitive-schema-default" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_loop_body_default_is_official_agentspec_json(loop_body) -> None:
    plan = _unwrap_executable(build_loop(loop_body))
    body_spec = _serialize_workflow(_wrap_executable(_bodies(plan)[0].agent))
    assert json.loads(body_spec)["component_type"] == "Agent"


def test_loop_accepts_direct_node_body_and_preserves_its_contract(database_llm) -> None:
    body = llm_node(
        name="review",
        llm=database_llm,
        prompt="Update {{done}}.",
        inputs={"done": "string"},
        outputs={"done": "string"},
    )
    source = AgentSpecSerializer().to_json(_unwrap_executable(body))
    plan = _unwrap_executable(build_loop(body))
    body_spec = _serialize_workflow(_wrap_executable(_bodies(plan)[0]))
    decoded = AgentSpecDeserializer().from_json(body_spec)
    assert isinstance(decoded, Flow)
    assert [(p.title, p.type) for p in decoded.inputs] == [("done", "string")]
    assert [(p.title, p.type) for p in decoded.outputs] == [("done", "string")]
    assert AgentSpecSerializer().to_json(_unwrap_executable(body)) == source


@pytest.mark.parametrize("bound", [1, 256])
def test_loop_accepts_valid_iteration_bound_endpoints(loop_body, bound) -> None:
    plan = _unwrap_executable(build_loop(loop_body, max_iterations=bound))
    assert len(_bodies(plan)) == bound
    assert len(_conditions(plan)) == bound + 1
    _assert_acyclic(plan)


def test_loop_rejects_nullable_control_contract(database_llm) -> None:
    contract = {"done": {"type": ["string", "null"]}}
    body = agent(
        name="step",
        llm=database_llm,
        instruction="Update {{done}}.",
        inputs=contract,
        outputs=contract,
    )
    with pytest.raises(IncompatibleLoopContractError, match="string body state field"):
        build_loop(body)


def _assert_status_guards(flow):
    """Every child status must be checked before its original continuation."""
    parent_field = status_field(flow)
    assert parent_field
    exhausted = next(
        node
        for node in flow.nodes
        if isinstance(node, EndNode)
        and any(
            prop.title == parent_field and prop.json_schema.get("default") == "exhausted"
            for prop in node.outputs
        )
    )
    for source in flow.nodes:
        children = source.subflows if isinstance(source, ParallelFlowNode) else [source]
        child_fields = [status_field(child) for child in children if status_field(child)]
        if not child_fields:
            continue
        outgoing = [edge for edge in flow.control_flow_connections if edge.from_node is source]
        assert outgoing
        checks = {edge.to_node.id for edge in outgoing}
        assert len(checks) == 1
        current = outgoing[0].to_node
        checked = set()
        while isinstance(current, BranchingNode) and current.mapping == {"exhausted": "exhausted"}:
            selector = current.inputs[0].title
            assert selector not in checked
            checked.add(selector)
            assert any(
                edge.source_node is source
                and edge.source_output == selector
                and edge.destination_node is current
                and edge.destination_input == selector
                for edge in flow.data_flow_connections
            )
            branches = {
                edge.from_branch: edge.to_node
                for edge in flow.control_flow_connections
                if edge.from_node is current
            }
            assert branches["exhausted"] is exhausted
            current = branches["default"]
        assert checked == set(child_fields)


def test_nested_loop_checks_child_exhaustion_before_next_iteration(loop_body):
    inner = build_loop(loop_body, max_iterations=2)
    outer = _unwrap_executable(build_loop(inner, max_iterations=3))
    _assert_acyclic(outer)
    _assert_status_guards(outer)
    assert len(_bodies(outer)) == 3
    for body in _bodies(outer):
        assert len(_bodies(body.subflow)) == 2
        assert status_field(body.subflow) != status_field(outer)


def test_sequence_exhaustion_needs_no_final_success_fields(loop_body, make_agent):
    repeating = build_loop(loop_body, max_iterations=2)
    final = make_agent(
        "report",
        inputs=(
            ("counter", "integer"),
            ("limit", "integer"),
            ("done", "string"),
            ("trace", "string"),
        ),
        outputs=(("report", "string"),),
    )
    graph = _unwrap_executable(sequential(name="report_sequence", steps=[repeating, final]))
    _assert_status_guards(graph)
    assert [prop.title for prop in public_outputs(graph)] == ["report"]
    final_output = next(prop for prop in graph.outputs if prop.title == "report")
    assert "default" in final_output.json_schema
    success = next(
        node for node in graph.nodes if isinstance(node, EndNode) and node.branch_name == "next"
    )
    assert (
        "default"
        not in next(prop for prop in success.outputs if prop.title == "report").json_schema
    )
    exhausted = next(
        node for node in graph.nodes if isinstance(node, EndNode) and node is not success
    )
    assert [prop.title for prop in exhausted.outputs] == [status_field(graph)]
    AgentSpecDeserializer().from_json(_serialize_workflow(_wrap_executable(graph)))


def test_parallel_independent_loops_check_both_unique_statuses(loop_body, make_agent):
    first = build_loop(loop_body, max_iterations=2)
    second = loop(
        name="second",
        body=make_agent(
            "second_body", inputs=(("other_done", "string"), ("other_value", "integer"))
        ),
        state_key="other_done",
        continue_value="N",
        max_iterations=2,
    )
    graph = _unwrap_executable(parallel(name="parallel_test", steps=[first, second]))
    worker = next(node for node in graph.nodes if isinstance(node, ParallelFlowNode))
    child_fields = [status_field(flow) for flow in worker.subflows]
    assert len(set(child_fields)) == 2
    assert status_field(graph) not in child_fields
    assert len(public_outputs(graph)) == 6
    _assert_status_guards(graph)
    _assert_acyclic(graph)
    AgentSpecDeserializer().from_json(_serialize_workflow(_wrap_executable(graph)))


def test_routing_loop_and_nonloop_share_public_contract(loop_body, database_llm):
    selector = agent(
        name="selector",
        llm=database_llm,
        instruction="Choose using {{counter}}, {{limit}}, {{done}}, {{trace}}.",
        inputs={prop.title: prop.json_schema for prop in _unwrap_executable(loop_body).inputs},
        outputs={"route": "string"},
    )
    graph = _unwrap_executable(
        routing(
            selector,
            name="routing_test",
            route_key="route",
            routes={"repeat": build_loop(loop_body, max_iterations=2)},
            default=loop_body,
        )
    )
    _assert_status_guards(graph)
    _assert_acyclic(graph)
    for node in graph.nodes:
        if isinstance(node, FlowNode) and len(node.branches) > 1:
            assert {
                edge.from_branch
                for edge in graph.control_flow_connections
                if edge.from_node is node
            } == set(node.branches)
    AgentSpecDeserializer().from_json(_serialize_workflow(_wrap_executable(graph)))


@pytest.mark.parametrize(
    "value_schema",
    [
        {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]},
        {"type": "array", "items": {"type": "integer"}},
        {"type": ["null", "string"]},
        {"type": "string", "enum": ["allowed"]},
        {"type": "integer", "default": 42},
    ],
)
def test_transport_defaults_preserve_public_and_body_contracts(database_llm, value_schema):
    contract = {"done": "string", "value": value_schema}
    body = agent(
        name="typed",
        llm=database_llm,
        instruction="Update {{done}} and {{value}}.",
        inputs=contract,
        outputs=contract,
    )
    original = _serialize_workflow(body)
    graph = _unwrap_executable(build_loop(body, max_iterations=1))
    before = _unwrap_executable(body)
    assert [prop.json_schema for prop in public_outputs(graph)] == [
        prop.json_schema for prop in before.outputs
    ]
    assert [prop.json_schema for prop in _bodies(graph)[0].outputs] == [
        prop.json_schema for prop in before.outputs
    ]
    assert _serialize_workflow(body) == original
    AgentSpecDeserializer().from_json(_serialize_workflow(_wrap_executable(graph)))


def test_generated_status_avoids_user_field_collision(database_llm, monkeypatch):
    from itertools import count
    from types import SimpleNamespace

    import oracle_db_agent_sdk._flow_status as flow_status

    identifiers = count()
    monkeypatch.setattr(
        flow_status, "uuid4", lambda: SimpleNamespace(hex="collision", nonce=next(identifiers))
    )
    contract = {"done": "string", "__sdk_loop_status_collision": "string"}
    body = agent(
        name="collision",
        llm=database_llm,
        instruction="Update {{done}} and {{__sdk_loop_status_collision}}.",
        inputs=contract,
        outputs=contract,
    )
    graph = _unwrap_executable(build_loop(body, max_iterations=1))
    assert status_field(graph) == "__sdk_loop_status_collision_"
    assert [prop.title for prop in public_outputs(graph)] == list(contract)
    assert len({prop.title for prop in graph.outputs}) == len(graph.outputs)
    AgentSpecDeserializer().from_json(_serialize_workflow(_wrap_executable(graph)))


def test_parallel_loop_exhaustion_blocks_later_sequence_work(loop_body, make_agent):
    repeating = build_loop(loop_body, max_iterations=1)
    independent = make_agent("independent", inputs=(), outputs=(("other", "string"),))
    workers = parallel(name="parallel_test", steps=[repeating, independent])
    final = make_agent(
        "final",
        inputs=(
            ("counter", "integer"),
            ("limit", "integer"),
            ("done", "string"),
            ("trace", "string"),
            ("other", "string"),
        ),
        outputs=(("summary", "string"),),
    )
    graph = _unwrap_executable(sequential(name="after_parallel", steps=[workers, final]))
    _assert_status_guards(graph)
    _assert_acyclic(graph)
    child = next(node for node in graph.nodes if isinstance(node, FlowNode))
    _assert_status_guards(child.subflow)
    AgentSpecDeserializer().from_json(_serialize_workflow(_wrap_executable(graph)))


@pytest.mark.parametrize(
    "pattern", ["sequential", "parallel", "routing", "sequential_wrapped", "parallel_wrapped"]
)
def test_composed_loop_preserves_nonloop_output_schema_in_public_metadata(
    loop_body, database_llm, pattern
):
    schema = {
        "type": "string",
        "anyOf": [{"type": "string", "maxLength": 10}],
        "default": "ok",
    }
    repeating = build_loop(loop_body, max_iterations=1)
    inputs = {prop.title: prop.json_schema for prop in _unwrap_executable(loop_body).inputs}
    final = agent(
        name="final",
        llm=database_llm,
        instruction="Read {{counter}}, {{limit}}, {{done}}, {{trace}}.",
        inputs=inputs,
        outputs={"report": schema},
    )
    expected = deepcopy(_unwrap_executable(final).outputs[0].json_schema)
    if pattern == "sequential":
        workflow = sequential(name="report", steps=[repeating, final])
    elif pattern == "parallel":
        workflow = parallel(name="parallel_test", steps=[repeating, final])
    elif pattern == "sequential_wrapped":
        inner = sequential(name="report", steps=[repeating, final])
        workflow = sequential(name="outer", steps=[loop_body, inner])
    elif pattern == "parallel_wrapped":
        inner = sequential(name="report", steps=[repeating, final])
        workflow = parallel(name="parallel_test", steps=[loop_body, inner])
    else:
        selector = agent(
            name="selector",
            llm=database_llm,
            instruction="Read {{counter}}, {{limit}}, {{done}}, {{trace}}.",
            inputs=inputs,
            outputs={"route": "string"},
        )
        workflow = routing(
            selector,
            name="routing_test",
            route_key="route",
            routes={"repeat": sequential(name="report", steps=[repeating, final])},
            default=final,
        )
    graph = _unwrap_executable(workflow)
    output = next(prop for prop in public_outputs(graph) if prop.title == "report")
    assert output.json_schema == expected
    for node in graph.nodes:
        if isinstance(node, EndNode):
            for prop in node.outputs:
                if prop.title == "report":
                    assert prop.json_schema == expected
    assert _unwrap_executable(final).outputs[0].json_schema == expected
