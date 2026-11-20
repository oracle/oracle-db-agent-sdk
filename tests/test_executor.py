##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for the public workflow-first execution API."""

import inspect
import json
import traceback
from types import SimpleNamespace

import pytest

import oracle_db_agent_sdk
import oracle_db_agent_sdk.executor as executor_module
from oracle_db_agent_sdk import (
    AgentSpecSerializationError,
    BuildingBlockError,
    DatabaseExecutionError,
    InputSerializationError,
    InvalidDatabaseResponseError,
    execute,
)
from oracle_db_agent_sdk._wrappers import _unwrap_executable


def test_public_execute_api_accepts_workflow_not_serialized_json() -> None:
    parameters = inspect.signature(execute).parameters

    assert list(parameters) == ["connection", "workflow", "inputs"]
    assert not hasattr(oracle_db_agent_sdk, "serialize")


def test_execute_serializes_workflow_and_inputs(make_agent, monkeypatch) -> None:
    captured = {}
    workflow = make_agent("support")
    source_json = ' { "component_type" : "Agent" } '

    def fake_serialize_workflow(value):
        captured["workflow"] = value
        return source_json

    def fake_execute_immediate(connection, *, agent_spec_json, inputs_json):
        captured.update(
            connection=connection,
            agent_spec_json=agent_spec_json,
            inputs_json=inputs_json,
        )
        return '{"answer":"café"}'

    monkeypatch.setattr(
        executor_module,
        "_serialize_workflow",
        fake_serialize_workflow,
    )
    monkeypatch.setattr(executor_module, "execute_immediate", fake_execute_immediate)
    connection = object()

    result = execute(
        connection=connection,  # type: ignore[arg-type]
        workflow=workflow,
        inputs={"question": "café"},
    )

    assert result == {"answer": "café"}
    assert captured == {
        "workflow": workflow,
        "connection": connection,
        "agent_spec_json": source_json,
        "inputs_json": '[{"name":"question","value":"café"}]',
    }


def test_execute_serializes_each_named_input_as_database_record(
    make_agent,
    monkeypatch,
) -> None:
    captured = {}

    def fake_execute_immediate(connection, *, agent_spec_json, inputs_json):
        captured["inputs_json"] = inputs_json
        return "null"

    monkeypatch.setattr(executor_module, "execute_immediate", fake_execute_immediate)

    execute(
        connection=object(),  # type: ignore[arg-type]
        workflow=make_agent("support"),
        inputs={"question": "Where is my order?", "attempt": 1},
    )

    assert captured["inputs_json"] == (
        '[{"name":"question","value":"Where is my order?"},{"name":"attempt","value":1}]'
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"value":1}', {"value": 1}),
        ("[1,2]", [1, 2]),
        ('"answer"', "answer"),
        ("42", 42),
        ("true", True),
        ("null", None),
        ("1.7976931348623157e308", 1.7976931348623157e308),
        ("-1.7976931348623157e308", -1.7976931348623157e308),
        ('{"value":[1.25e2,-2.5e-2]}', {"value": [125.0, -0.025]}),
    ],
)
def test_execute_returns_any_valid_json_value(
    make_agent,
    monkeypatch,
    raw,
    expected,
) -> None:
    monkeypatch.setattr(executor_module, "execute_immediate", lambda *args, **kwargs: raw)

    assert (
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={},
        )
        == expected
    )


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_missing_database_response_is_rejected(make_agent, monkeypatch, raw) -> None:
    monkeypatch.setattr(executor_module, "execute_immediate", lambda *args, **kwargs: raw)

    with pytest.raises(InvalidDatabaseResponseError, match="returned no agent response"):
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={},
        )


def test_non_text_database_response_is_rejected(make_agent, monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "execute_immediate", lambda *args, **kwargs: b"{}")

    with pytest.raises(InvalidDatabaseResponseError, match="must be JSON text"):
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={},
        )


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "NaN",
        "Infinity",
        "-Infinity",
        "1e400",
        "-1e400",
        '{"value":1e400}',
        '{"value":-1e400}',
        '[{"value":[1e400]}]',
        '[{"value":[-1e400]}]',
    ],
)
def test_malformed_database_response_is_rejected(make_agent, monkeypatch, raw) -> None:
    monkeypatch.setattr(executor_module, "execute_immediate", lambda *args, **kwargs: raw)

    with pytest.raises(InvalidDatabaseResponseError, match="is not valid JSON"):
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={},
        )


def test_input_serialization_fails_before_database_call(make_agent, monkeypatch) -> None:
    called = False

    def should_not_execute(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(executor_module, "execute_immediate", should_not_execute)

    with pytest.raises(
        InputSerializationError,
        match="Execution input 'unsupported' is not JSON serializable",
    ):
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={"unsupported": object()},  # type: ignore[dict-item]
        )

    assert not called


@pytest.mark.parametrize("kind", ["array", "object"])
def test_input_encoder_recursion_error_uses_sdk_error_before_database_call(
    make_agent, monkeypatch, kind
) -> None:
    payload = [0] if kind == "array" else {"nested": 0}
    calls = []

    def encode(value, *args, **kwargs):
        if value is payload:
            raise RecursionError("Encoder nesting limit reached")
        return json.dumps(value, *args, **kwargs)

    monkeypatch.setattr(executor_module, "json", SimpleNamespace(dumps=encode))
    monkeypatch.setattr(executor_module, "execute_immediate", lambda *a, **k: calls.append(k))

    with pytest.raises(InputSerializationError) as caught:
        execute(connection=object(), workflow=make_agent("support"), inputs={"payload": payload})

    assert not calls
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("kind", ["array", "object"])
def test_response_decoder_recursion_error_uses_sdk_error_after_one_call(
    make_agent, monkeypatch, kind
) -> None:
    raw = json.dumps([0] if kind == "array" else {"nested": 0})
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return raw

    def decode(value, *args, **kwargs):
        assert value == raw
        raise RecursionError("Decoder nesting limit reached")

    monkeypatch.setattr(
        executor_module,
        "json",
        SimpleNamespace(dumps=json.dumps, loads=decode, JSONDecodeError=json.JSONDecodeError),
    )
    monkeypatch.setattr(executor_module, "execute_immediate", run)
    with pytest.raises(InvalidDatabaseResponseError) as caught:
        execute(connection=object(), workflow=make_agent("support"), inputs={})

    assert len(calls) == 1
    assert caught.value.__cause__ is None


def test_input_envelope_nesting_failure_uses_sdk_error(monkeypatch) -> None:
    payload = {"nested": [0]}

    def encode(value, *args, **kwargs):
        if value == [{"name": "payload", "value": payload}]:
            raise RecursionError("Envelope nesting limit reached")
        return json.dumps(value, *args, **kwargs)

    monkeypatch.setattr(executor_module, "json", SimpleNamespace(dumps=encode))
    with pytest.raises(InputSerializationError) as caught:
        executor_module._serialize_inputs({"payload": payload})
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("kind", ["array", "object"])
def test_nested_json_values_round_trip_with_real_codec(make_agent, monkeypatch, kind) -> None:
    payload = 0
    for _ in range(64):
        payload = [payload] if kind == "array" else {"nested": payload}
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        assert json.loads(kwargs["inputs_json"]) == [{"name": "payload", "value": payload}]
        return json.dumps(payload)

    monkeypatch.setattr(executor_module, "execute_immediate", run)
    assert (
        execute(connection=object(), workflow=make_agent("support"), inputs={"payload": payload})
        == payload
    )
    assert len(calls) == 1


def test_input_serialization_error_details_are_absent_from_traceback(make_agent) -> None:
    sensitive_value = "synthetic-sensitive-value"

    class InvalidInput(dict):
        def items(self):
            raise ValueError(sensitive_value)

    with pytest.raises(InputSerializationError) as caught:
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={"value": InvalidInput(value=sensitive_value)},
        )

    assert sensitive_value not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None


def test_response_parse_error_details_are_absent_from_traceback(make_agent, monkeypatch) -> None:
    sensitive_value = "synthetic-sensitive-value"
    original = ValueError(sensitive_value)

    def fail_decode(*args, **kwargs):
        raise original

    monkeypatch.setattr(executor_module, "execute_immediate", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(
        executor_module,
        "json",
        SimpleNamespace(
            loads=fail_decode,
            dumps=executor_module.json.dumps,
            JSONDecodeError=executor_module.json.JSONDecodeError,
        ),
    )

    with pytest.raises(InvalidDatabaseResponseError) as caught:
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={},
        )

    assert sensitive_value not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None


def test_execute_rejects_non_mapping_inputs_before_database_call(
    make_agent,
    monkeypatch,
) -> None:
    called = False

    def should_not_execute(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(executor_module, "execute_immediate", should_not_execute)

    with pytest.raises(InputSerializationError, match="inputs must be a mapping"):
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs=None,  # type: ignore[arg-type]
        )

    assert not called


def test_execute_rejects_non_string_input_name_before_database_call(
    make_agent,
    monkeypatch,
) -> None:
    called = False

    def should_not_execute(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(executor_module, "execute_immediate", should_not_execute)

    with pytest.raises(InputSerializationError, match="input names must be strings"):
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={1: "value"},  # type: ignore[dict-item]
        )

    assert not called


def test_workflow_serialization_fails_before_database_call(make_agent, monkeypatch) -> None:
    original = AgentSpecSerializationError("AgentSpec serialization failed.")
    called = False

    def fail_workflow_serialization(workflow):
        raise original

    def should_not_execute(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        executor_module,
        "_serialize_workflow",
        fail_workflow_serialization,
    )
    monkeypatch.setattr(executor_module, "execute_immediate", should_not_execute)

    with pytest.raises(AgentSpecSerializationError) as caught:
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={},
        )

    assert caught.value is original
    assert not called


def test_execute_rejects_raw_pyagentspec_before_database_call(make_agent, monkeypatch) -> None:
    called = False

    def should_not_execute(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(executor_module, "execute_immediate", should_not_execute)
    raw_agent = _unwrap_executable(make_agent("support"))

    with pytest.raises(BuildingBlockError, match="workflow must be created"):
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=raw_agent,  # type: ignore[arg-type]
            inputs={},
        )

    assert not called


def test_database_error_is_preserved(make_agent, monkeypatch) -> None:
    original = RuntimeError("driver detail")
    wrapped = DatabaseExecutionError("Database agent execution failed.")
    wrapped.__cause__ = original

    def fail(*args, **kwargs):
        raise wrapped

    monkeypatch.setattr(executor_module, "execute_immediate", fail)

    with pytest.raises(DatabaseExecutionError) as caught:
        execute(
            connection=object(),  # type: ignore[arg-type]
            workflow=make_agent("support"),
            inputs={},
        )

    assert caught.value is wrapped
