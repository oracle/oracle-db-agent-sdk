##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Opt-in database tests that create and remove isolated test resources.

Set SDK_FIXED_LOOP_CONNECTION_FILE to a protected JSON file containing user,
password, and dsn. Set SDK_FIXED_LOOP_CLIENT_LIB if thick mode is required.
The target schema needs CREATE TABLE/PROCEDURE and agent execution privileges.
No model credentials, proxy changes, or network ACL changes are needed.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import oracledb
import pytest

import oracle_db_agent_sdk.executor as sdk_executor
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
from oracle_db_agent_sdk import sequential as bounded_sequence
from oracle_db_agent_sdk._wrappers import _unwrap_executable

CONNECTION_FILE = os.environ.get("SDK_FIXED_LOOP_CONNECTION_FILE")
pytestmark = pytest.mark.skipif(not CONNECTION_FILE, reason="live database is opt-in")
FIELDS = {"counter": "number", "target": "number", "done": "string", "action_kind": "string"}
NEXT_FIELDS = {f"next_{key}": value for key, value in FIELDS.items()}


@pytest.fixture(scope="module")
def connection():
    config = json.loads(Path(CONNECTION_FILE).read_text())
    if lib_dir := os.environ.get("SDK_FIXED_LOOP_CLIENT_LIB"):
        oracledb.init_oracle_client(lib_dir=lib_dir)
    with oracledb.connect(**{key: config[key] for key in ("user", "password", "dsn")}) as conn:
        conn.call_timeout = 60000
        yield conn
        conn.rollback()


@pytest.fixture(scope="module")
def resources(connection):
    suffix = uuid4().hex[:12].upper()
    package, events, data = (f"SDK_UL_{prefix}_{suffix}" for prefix in ("P", "E", "D"))
    created = []
    with connection.cursor() as cursor:
        try:
            for table in (events, data):
                cursor.execute(f"create table {table} (value number)")
                created.append(("table", table))
            cursor.execute(f"""
                create package {package} authid current_user as
                  procedure step(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, next_counter out number, next_target out number,
                    next_done out varchar2, next_action_kind out varchar2);
                  procedure rename(next_counter in number, next_target in number,
                    next_done in varchar2, next_action_kind in varchar2,
                    counter out number, target out number, done out varchar2,
                    action_kind out varchar2);
                  procedure reset_state(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, next_counter out number, next_target out number,
                    next_done out varchar2, next_action_kind out varchar2);
                  procedure choose(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, route out varchar2);
                  procedure other(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, other_value out number);
                  procedure summarize(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, summary out varchar2);
                  procedure second_step(turns in number, stop_at in number, more in varchar2,
                    next_turns out number, next_stop_at out number, next_more out varchar2);
                  procedure second_rename(next_turns in number, next_stop_at in number,
                    next_more in varchar2, turns out number, stop_at out number, more out varchar2);
                end;
            """)
            created.append(("package", package))
            cursor.execute(f"""
                create package body {package} as
                  procedure event(value number) is
                    pragma autonomous_transaction;
                  begin insert into {events} values(value); commit; end;
                  procedure step(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, next_counter out number, next_target out number,
                    next_done out varchar2, next_action_kind out varchar2) is
                  begin
                    event(counter);
                    if action_kind in ('FAIL','FAIL_DML') and counter>=1 then
                      raise_application_error(-20991, 'fixed-length loop test body failure');
                    end if;
                    if action_kind in ('DML','FAIL_DML') then
                      insert into {data} values(counter);
                    end if;
                    next_counter:=counter+1; next_target:=target;
                    next_action_kind:=case when action_kind like 'TEXT:%'
                      then action_kind||'['||to_char(next_counter)||']'
                      else action_kind end;
                    next_done:=case when next_counter>=target then 'Y' else 'N' end;
                  end;
                  procedure rename(next_counter in number, next_target in number,
                    next_done in varchar2, next_action_kind in varchar2,
                    counter out number, target out number, done out varchar2,
                    action_kind out varchar2) is
                  begin counter:=next_counter; target:=next_target;
                    done:=next_done; action_kind:=next_action_kind; end;
                  procedure reset_state(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, next_counter out number, next_target out number,
                    next_done out varchar2, next_action_kind out varchar2) is
                  begin next_counter:=counter; next_target:=target+2;
                    next_done:='N'; next_action_kind:=action_kind; end;
                  procedure choose(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, route out varchar2) is
                  begin route:=action_kind; end;
                  procedure other(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, other_value out number) is
                  begin other_value:=counter+100; end;
                  procedure summarize(counter in number, target in number, done in varchar2,
                    action_kind in varchar2, summary out varchar2) is
                  begin event(-1000); summary:='result '||to_char(counter); end;
                  procedure second_step(turns in number, stop_at in number, more in varchar2,
                    next_turns out number, next_stop_at out number, next_more out varchar2) is
                  begin
                    event(10000+turns); next_turns:=turns+1; next_stop_at:=stop_at;
                    next_more:=case when next_turns>=stop_at then 'Y' else 'N' end;
                  end;
                  procedure second_rename(next_turns in number, next_stop_at in number,
                    next_more in varchar2, turns out number, stop_at out number,
                    more out varchar2) is
                  begin turns:=next_turns; stop_at:=next_stop_at; more:=next_more; end;
                end;
            """)
            errors = cursor.execute(
                "select line, position, text from user_errors where name=:name order by sequence",
                name=package,
            ).fetchall()
            assert not errors, errors
            yield package, events, data
        finally:
            connection.rollback()
            for kind, name in reversed(created):
                cursor.execute(f"drop {kind} {name}" + (" purge" if kind == "table" else ""))


@pytest.fixture
def body(connection, resources):
    package, events, data = resources
    with connection.cursor() as cursor:
        cursor.execute(f"delete from {events}")
        cursor.execute(f"delete from {data}")
    connection.commit()
    return sequential(
        name="loop-body",
        steps=[
            node(package, "step", FIELDS, NEXT_FIELDS),
            node(package, "rename", NEXT_FIELDS, FIELDS),
        ],
    )


def node(package, procedure, inputs, outputs):
    return tool_node(
        name=procedure,
        tool=plsql_tool(
            name=procedure,
            procedure=procedure,
            package=package,
            inputs=inputs,
            outputs=outputs,
        ),
    )


def repeated(body, bound=3):
    return loop(
        name="fixed-repeat", body=body, state_key="done", continue_value="N", max_iterations=bound
    )


def values(counter=0, target=2, done="N", action_kind="NORMAL"):
    return dict(counter=counter, target=target, done=done, action_kind=action_kind)


def count_events(connection, resources):
    with connection.cursor() as cursor:
        return cursor.execute(f"select count(*) from {resources[1]}").fetchone()[0]


@pytest.mark.parametrize(
    ("initial", "bound", "expected", "calls"),
    [
        (values(done="Y"), 1, values(done="Y"), 0),
        (values(target=1), 1, values(counter=1, target=1, done="Y"), 1),
        (values(target=3), 3, values(counter=3, target=3, done="Y"), 3),
        (values(counter=2, target=4), 5, values(counter=4, target=4, done="Y"), 2),
    ],
)
def test_fixed_feedback(connection, resources, body, initial, bound, expected, calls):
    assert (
        execute(connection=connection, workflow=repeated(body, bound), inputs=initial) == expected
    )
    assert count_events(connection, resources) == calls


def test_bound_prevents_an_extra_body_execution(connection, resources, body):
    with pytest.raises(LoopExecutionError, match="iteration limit"):
        execute(connection=connection, workflow=repeated(body, 1), inputs=values(target=2))
    assert count_events(connection, resources) == 1


def test_body_failure_is_not_relabelled_as_limit(connection, resources, body):
    with pytest.raises((DatabaseExecutionError, InvalidDatabaseResponseError)) as caught:
        execute(connection=connection, workflow=repeated(body), inputs=values(action_kind="FAIL"))
    assert "Database loop iteration limit reached" not in str(caught.value)
    assert count_events(connection, resources) == 2


@pytest.mark.parametrize("kind", ["sequential", "nested", "route", "default", "parallel"])
def test_loop_composition(connection, resources, body, kind):
    repeating = repeated(body)
    expected = values(counter=2, done="Y")
    inputs = values()
    if kind == "sequential":
        workflow = bounded_sequence(name="reuse", steps=[repeating, repeating])
    elif kind == "nested":
        workflow = repeated(repeating)
    elif kind in {"route", "default"}:
        selector = node(resources[0], "choose", FIELDS, {"route": "string"})
        workflow = routing(
            selector,
            name="routing_test",
            route_key="route",
            routes={"NORMAL": repeating},
            default=repeated(body, 2),
        )
        if kind == "default":
            inputs["action_kind"] = expected["action_kind"] = "UNKNOWN"
    else:
        other = node(resources[0], "other", FIELDS, {"other_value": "number"})
        workflow = parallel(name="parallel_test", steps=[repeating, other])
        expected["other_value"] = 100
    assert execute(connection=connection, workflow=workflow, inputs=inputs) == expected
    assert count_events(connection, resources) == 2


def test_fixed_loop_leaves_transaction_control_to_caller(connection, resources, body):
    with connection.cursor() as cursor:
        cursor.execute(f"insert into {resources[2]} values(-1)")
    result = execute(
        connection=connection, workflow=repeated(body), inputs=values(action_kind="DML")
    )
    assert result == values(counter=2, done="Y", action_kind="DML")
    with connection.cursor() as cursor:
        assert cursor.execute(f"select count(*) from {resources[2]}").fetchone()[0] == 3
        connection.rollback()
        assert cursor.execute(f"select count(*) from {resources[2]}").fetchone()[0] == 0


def test_reused_loop_resets_budget_for_each_entry(connection, resources, body):
    repeating = repeated(body, 2)
    reset = sequential(
        name="reset",
        steps=[
            node(resources[0], "reset_state", FIELDS, NEXT_FIELDS),
            node(resources[0], "rename", NEXT_FIELDS, FIELDS),
        ],
    )
    workflow = bounded_sequence(name="repeat-reset-repeat", steps=[repeating, reset, repeating])
    result = execute(connection=connection, workflow=workflow, inputs=values())
    assert result == values(counter=4, target=4, done="Y")
    assert count_events(connection, resources) == 4


def test_failure_preserves_preexisting_caller_transaction(
    connection, resources, body, record_property
):
    with connection.cursor() as cursor:
        cursor.execute(f"insert into {resources[2]} values(-1)")
    with pytest.raises((DatabaseExecutionError, InvalidDatabaseResponseError)):
        execute(
            connection=connection,
            workflow=repeated(body),
            inputs=values(target=3, action_kind="FAIL_DML"),
        )
    with connection.cursor() as cursor:
        rows = cursor.execute(f"select value from {resources[2]} order by value").fetchall()
        assert (-1,) in rows
        # Record Oracle statement-level failure behavior without assuming that
        # earlier iterations inside the single database call survive its error.
        record_property("in_call_iteration_rows_after_failure", sum(row[0] >= 0 for row in rows))
        connection.rollback()
        assert cursor.execute(f"select count(*) from {resources[2]}").fetchone()[0] == 0
    assert count_events(connection, resources) == 2


ARTIFACTS = Path(os.environ.get("SDK_FIXED_LOOP_ARTIFACTS", "/tmp/sdk-fixed-loop/artifacts"))
ARTIFACTS.mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def capture_calls(monkeypatch, request):
    original = sdk_executor.execute_immediate
    calls = []

    def observed(connection, *, agent_spec_json, inputs_json):
        assert "agent_sql_pattern_loop" not in agent_spec_json
        assert "loop_gate_id" not in agent_spec_json
        before = time.perf_counter()
        slug = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:12]
        (ARTIFACTS / f"{slug}.spec.json").write_text(agent_spec_json)
        record = {"bytes": len(agent_spec_json.encode()), "inputs": json.loads(inputs_json)}
        calls.append(record)
        try:
            result = original(connection, agent_spec_json=agent_spec_json, inputs_json=inputs_json)
            record["result"] = json.loads(result) if result else None
            return result
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["seconds"] = time.perf_counter() - before

    monkeypatch.setattr(sdk_executor, "execute_immediate", observed)
    yield calls
    assert len(calls) <= 1, "SDK must submit once, never iterate in Python"
    slug = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:12]
    (ARTIFACTS / f"{slug}.result.json").write_text(
        json.dumps({"case": request.node.nodeid, "calls": calls}, indent=2)
    )


@pytest.mark.parametrize("bound", [1, 3, 5, 10])
def test_always_continue_is_bounded(connection, resources, body, bound):
    workflow = repeated(body, bound)
    graph = _unwrap_executable(workflow)
    assert sum(n.name.startswith("loop-body-") for n in graph.nodes) == bound
    with pytest.raises(LoopExecutionError):
        execute(connection=connection, workflow=workflow, inputs=values(target=1000))
    assert count_events(connection, resources) == bound


def test_exhaustion_skips_later_sequence_steps(connection, resources, body):
    workflow = bounded_sequence(name="stop-on-exhaustion", steps=[repeated(body, 1), body])
    with pytest.raises(LoopExecutionError):
        execute(connection=connection, workflow=workflow, inputs=values(target=3))
    assert count_events(connection, resources) == 1


def test_inner_exhaustion_skips_outer_body_and_future_iterations(connection, resources, body):
    inner = repeated(body, 1)
    outer_body = bounded_sequence(name="inner-and-later", steps=[inner, body])
    workflow = repeated(outer_body, 3)
    with pytest.raises(LoopExecutionError):
        execute(connection=connection, workflow=workflow, inputs=values(target=10))
    assert count_events(connection, resources) == 1


def test_nested_exhaustion_final_state(connection, resources, body):
    workflow = repeated(repeated(body, 2), 3)
    with pytest.raises(LoopExecutionError):
        execute(connection=connection, workflow=workflow, inputs=values(target=10))
    assert count_events(connection, resources) == 2


def test_parallel_reports_child_exhaustion(connection, resources, body):
    other = node(resources[0], "other", FIELDS, {"other_value": "number"})
    workflow = parallel(name="parallel_test", steps=[repeated(body, 1), other])
    with pytest.raises(LoopExecutionError):
        execute(connection=connection, workflow=workflow, inputs=values(target=10))
    assert count_events(connection, resources) == 1


def test_limit_status_keeps_transaction_until_caller_rollback(connection, resources, body):
    with connection.cursor() as cursor:
        cursor.execute(f"insert into {resources[2]} values(-1)")
    with pytest.raises(LoopExecutionError):
        execute(
            connection=connection,
            workflow=repeated(body, 1),
            inputs=values(target=10, action_kind="DML"),
        )
    with connection.cursor() as cursor:
        assert cursor.execute(f"select value from {resources[2]} order by value").fetchall() == [
            (-1,),
            (0,),
        ]
        connection.rollback()
        assert cursor.execute(f"select count(*) from {resources[2]}").fetchone()[0] == 0
    assert count_events(connection, resources) == 1


@pytest.mark.parametrize("bound", [1, 5, 10, 32])
def test_zero_iteration_parse_scale(connection, resources, body, bound):
    result = execute(connection=connection, workflow=repeated(body, bound), inputs=values(done="Y"))
    assert result == values(done="Y")
    assert count_events(connection, resources) == 0


@pytest.mark.parametrize("route_value", ["NORMAL", "UNKNOWN"])
def test_routing_exhaustion_skips_later_step(connection, resources, body, route_value):
    router = node(resources[0], "choose", FIELDS, {"route": "string"})
    selected = routing(
        router,
        name="routing_test",
        route_key="route",
        routes={"NORMAL": repeated(body, 1)},
        default=repeated(body, 2),
    )
    workflow = bounded_sequence(name="route-then-body", steps=[selected, body])
    with pytest.raises(LoopExecutionError):
        execute(
            connection=connection,
            workflow=workflow,
            inputs=values(target=10, action_kind=route_value),
        )
    assert count_events(connection, resources) == (1 if route_value == "NORMAL" else 2)


def test_unicode_text_feedback(connection, resources, body):
    initial = values(target=3, action_kind="TEXT:café 你好")
    result = execute(connection=connection, workflow=repeated(body, 5), inputs=initial)
    assert result == values(counter=3, target=3, done="Y", action_kind="TEXT:café 你好[1][2][3]")
    assert initial == values(target=3, action_kind="TEXT:café 你好")
    assert count_events(connection, resources) == 3


@pytest.mark.parametrize("target", [1, 3])
def test_sequence_with_different_final_schema(connection, resources, body, target):
    summary = node(resources[0], "summarize", FIELDS, {"summary": "string"})
    workflow = sequential(name="review-and-summarize", steps=[repeated(body, 1), summary])
    if target == 1:
        assert execute(connection=connection, workflow=workflow, inputs=values(target=target)) == {
            "summary": "result 1"
        }
        assert count_events(connection, resources) == 2
    else:
        with pytest.raises(LoopExecutionError):
            execute(connection=connection, workflow=workflow, inputs=values(target=target))
        assert count_events(connection, resources) == 1


@pytest.mark.parametrize("first_target,second_target", [(2, 2), (3, 2), (2, 3), (3, 3)])
def test_two_parallel_loops(connection, resources, body, first_target, second_target):
    fields = {"turns": "number", "stop_at": "number", "more": "string"}
    next_fields = {f"next_{key}": value for key, value in fields.items()}
    second_body = sequential(
        name="second-body",
        steps=[
            node(resources[0], "second_step", fields, next_fields),
            node(resources[0], "second_rename", next_fields, fields),
        ],
    )
    second = loop(
        name="second-loop", body=second_body, state_key="more", continue_value="N", max_iterations=2
    )
    workflow = parallel(name="parallel_test", steps=[repeated(body, 2), second])
    inputs = {**values(target=first_target), "turns": 0, "stop_at": second_target, "more": "N"}
    if first_target == second_target == 2:
        assert execute(connection=connection, workflow=workflow, inputs=inputs) == {
            **values(counter=2, target=2, done="Y"),
            "turns": 2,
            "stop_at": 2,
            "more": "Y",
        }
    else:
        with pytest.raises(LoopExecutionError):
            execute(connection=connection, workflow=workflow, inputs=inputs)
    assert count_events(connection, resources) == 4


def constant_flow(name, inputs, outputs, result):
    """A deterministic database step, with required public outputs and End defaults."""
    from pyagentspec.flows.edges import ControlFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.nodes import EndNode, StartNode

    from oracle_db_agent_sdk._contracts import contract_properties
    from oracle_db_agent_sdk._wrappers import _wrap_executable

    start = StartNode(name=f"{name}-start", inputs=contract_properties(inputs))
    end_fields = contract_properties(outputs)
    for prop in end_fields:
        prop.json_schema = {**prop.json_schema, "default": result[prop.title]}
    end = EndNode(name=f"{name}-end", outputs=end_fields)
    return _wrap_executable(
        Flow(
            name=name,
            start_node=start,
            nodes=[start, end],
            outputs=contract_properties(outputs),
            control_flow_connections=[ControlFlowEdge(name="finish", from_node=start, to_node=end)],
            data_flow_connections=[],
        )
    )


@pytest.mark.parametrize("exhausts", [False, True])
@pytest.mark.parametrize(
    "schema,value",
    [
        ({"type": "object"}, {"items": [1, 2]}),
        ({"type": "number"}, 12),
        ({"type": "boolean"}, True),
        ({"type": "string"}, "DONE"),
    ],
)
def test_exhaustion_transport_with_different_final_types(
    connection, resources, body, exhausts, schema, value
):
    final = constant_flow("final", FIELDS, {"answer": schema}, {"answer": value})
    workflow = sequential(name="different-schema", steps=[repeated(body, 1), final])
    if exhausts:
        with pytest.raises(LoopExecutionError):
            execute(connection=connection, workflow=workflow, inputs=values(target=2))
    else:
        assert execute(connection=connection, workflow=workflow, inputs=values(target=1)) == {
            "answer": value
        }
    assert count_events(connection, resources) == 1


@pytest.mark.parametrize("exhausts", [False, True])
def test_router_can_itself_contain_a_loop(connection, resources, body, exhausts):
    fields = {**FIELDS, "route": "string"}
    router_body = constant_flow(
        "router-body", fields, fields, {**values(done="N" if exhausts else "Y"), "route": "review"}
    )
    router = repeated(router_body, 1)
    workflow = routing(
        router,
        name="routing_test",
        route_key="route",
        routes={"review": repeated(body)},
        default=repeated(body),
    )
    if exhausts:
        with pytest.raises(LoopExecutionError):
            execute(
                connection=connection, workflow=workflow, inputs={**values(), "route": "review"}
            )
        assert count_events(connection, resources) == 0
    else:
        assert execute(
            connection=connection, workflow=workflow, inputs={**values(), "route": "review"}
        ) == values(counter=2, done="Y")
        assert count_events(connection, resources) == 2
