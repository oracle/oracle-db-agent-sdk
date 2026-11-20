##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Verify bounded graph submission and final-result error boundaries.

These fake-driver tests prove client behavior, not database loop execution.
"""

import json
from copy import deepcopy

import pytest
from pyagentspec import AgentSpecDeserializer
from pyagentspec.flows.flow import Flow

import oracle_db_agent_sdk.executor as executor
from oracle_db_agent_sdk import (
    DatabaseExecutionError,
    InvalidDatabaseResponseError,
    LoopExecutionError,
    execute,
    loop,
    parallel,
    plsql_tool,
    routing,
    sequential,
    tool_node,
)
from oracle_db_agent_sdk._flow_status import status_field
from oracle_db_agent_sdk._serialization import _serialize_workflow
from oracle_db_agent_sdk._wrappers import _unwrap_executable


def step(name="body", inputs=None, outputs=None):
    contract = {"counter": "integer", "done": "string"} if inputs is None else inputs
    return tool_node(
        name=name,
        tool=plsql_tool(
            name=name,
            procedure=name,
            inputs=contract,
            outputs=contract if outputs is None else outputs,
        ),
    )


def repeat(body=None, bound=3):
    return loop(
        name="repeat",
        body=body or step(),
        state_key="done",
        continue_value="N",
        max_iterations=bound,
    )


def _compose(pattern):
    repeating = repeat()
    if pattern == "sequential":
        return sequential(name="twice", steps=[repeating, repeating])
    if pattern == "nested":
        return repeat(body=sequential(name="nested", steps=[repeating, step()]))
    if pattern == "routing":
        router = step(name="route", outputs={"route": "string"})
        return routing(
            router,
            name="routing_test",
            route_key="route",
            routes={"a": repeating},
            default=repeating,
        )
    if pattern == "parallel":
        return parallel(
            name="parallel_test", steps=[repeating, step(name="other", outputs={"other": "string"})]
        )
    if pattern == "two_parallel_loops":
        other = loop(
            name="other_loop",
            body=step(name="other", inputs={"other_done": "string", "value": "integer"}),
            state_key="other_done",
            continue_value="N",
            max_iterations=2,
        )
        return parallel(name="parallel_test", steps=[repeating, other])
    return repeating


@pytest.mark.parametrize(
    "pattern", ["loop", "sequential", "routing", "parallel", "nested", "two_parallel_loops"]
)
@pytest.mark.parametrize("status", ["complete", "exhausted"])
def test_loop_and_compositions_submit_once_and_classify_final_status(monkeypatch, pattern, status):
    calls = []
    connection = object()
    workflow = _compose(pattern)
    field = status_field(_unwrap_executable(workflow))
    assert field
    inputs = {"counter": 0, "done": "N"}
    if pattern == "two_parallel_loops":
        inputs.update(other_done="N", value=0)
    before = deepcopy(inputs)
    payload = {"counter": 2, "done": "Y", "__sdk_loop_status_user": "keep"}

    def run(c, *, agent_spec_json, inputs_json):
        calls.append(c)
        graph = AgentSpecDeserializer().from_json(agent_spec_json)
        assert isinstance(graph, Flow)
        assert status_field(graph) == field
        assert "agent_sql_pattern_loop" not in agent_spec_json
        assert "loop_gate_id" not in agent_spec_json
        assert {p["name"]: p["value"] for p in json.loads(inputs_json)} == inputs
        return json.dumps({**payload, field: status})

    monkeypatch.setattr(executor, "execute_immediate", run)
    if status == "complete":
        assert execute(connection=connection, workflow=workflow, inputs=inputs) == payload
    else:
        with pytest.raises(LoopExecutionError, match="iteration limit"):
            execute(connection=connection, workflow=workflow, inputs=inputs)
    assert calls == [connection]
    assert inputs == before


@pytest.mark.parametrize("message", ["body failed (ORA-50741).", "body failed (ORA-20991)."])
def test_database_errors_preserve_original_exception_without_retry(monkeypatch, message):
    calls = []
    error = DatabaseExecutionError(message)

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(executor, "execute_immediate", fail)
    with pytest.raises(DatabaseExecutionError) as caught:
        execute(connection=object(), workflow=repeat(), inputs={"counter": 0, "done": "N"})
    assert caught.value is error
    assert len(calls) == 1


@pytest.mark.parametrize(
    "raw",
    [None, "", "  ", "not-json", "null", "[]", "1", '"complete"', "{}", '{"done":"Y"}'],
)
def test_missing_or_invalid_response_is_not_loop_exhaustion(monkeypatch, raw):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(executor, "execute_immediate", run)
    with pytest.raises(InvalidDatabaseResponseError):
        execute(connection=object(), workflow=repeat(), inputs={"counter": 0, "done": "N"})
    assert len(calls) == 1


@pytest.mark.parametrize("value", [None, "unknown", "", 0, True, [], {}, "COMPLETE"])
def test_unrecognized_status_fails_closed(monkeypatch, value):
    workflow = repeat()
    field = status_field(_unwrap_executable(workflow))
    monkeypatch.setattr(
        executor,
        "execute_immediate",
        lambda *args, **kwargs: json.dumps({"counter": 0, "done": "Y", field: value}),
    )
    with pytest.raises(InvalidDatabaseResponseError, match="completion status"):
        execute(connection=object(), workflow=workflow, inputs={"counter": 0, "done": "N"})


def test_other_workflows_preserve_status_like_user_output(monkeypatch):
    payload = {"__sdk_loop_status_user": "exhausted", "done": "Y"}
    workflow = step(outputs={"__sdk_loop_status_user": "string", "done": "string"})
    monkeypatch.setattr(executor, "execute_immediate", lambda *a, **k: json.dumps(payload))
    assert (
        execute(connection=object(), workflow=workflow, inputs={"counter": 0, "done": "N"})
        == payload
    )


def test_reusing_workflow_has_no_execution_state_or_retries(monkeypatch):
    workflow = repeat()
    original = _serialize_workflow(workflow)
    field = status_field(_unwrap_executable(workflow))
    calls = []
    responses = iter(["exhausted", "complete"])

    def run(*args, **kwargs):
        calls.append(kwargs)
        return json.dumps({"counter": 1, "done": "Y", field: next(responses)})

    monkeypatch.setattr(executor, "execute_immediate", run)
    with pytest.raises(LoopExecutionError):
        execute(connection=object(), workflow=workflow, inputs={"counter": 0, "done": "N"})
    assert execute(connection=object(), workflow=workflow, inputs={"counter": 0, "done": "N"}) == {
        "counter": 1,
        "done": "Y",
    }
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert _serialize_workflow(workflow) == original
