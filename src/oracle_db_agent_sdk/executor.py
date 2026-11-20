##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Public workflow execution API."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import cast

import oracledb

from ._database_adapter import execute_immediate
from ._flow_status import failure_statuses, status_field
from ._json_types import JSONValue
from ._serialization import _serialize_workflow
from ._wrappers import Executable, _unwrap_executable
from .exceptions import (
    InputSerializationError,
    InvalidDatabaseResponseError,
    LoopExecutionError,
    OrchestrationExecutionError,
)


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN and infinity tokens accepted by Python's default JSON decoder."""
    raise ValueError(f"Non-standard JSON constant: {value}")


def _finite_float(value: str) -> float:
    """Reject JSON numbers whose conversion would overflow to infinity."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Database agent response contains a non-finite number.")
    return result


def _serialize_inputs(inputs: Mapping[str, JSONValue]) -> str:
    """Encode named inputs as the database's JSON list of name/value records.

    Validate serialization and top-level names; do not perform workflow schema
    validation. Non-finite numeric input values are rejected.
    """
    if not isinstance(inputs, Mapping):
        raise InputSerializationError(
            "inputs must be a mapping of string names to JSON-compatible values."
        )

    database_inputs: list[dict[str, JSONValue | str]] = []
    for name, value in inputs.items():
        if not isinstance(name, str):
            raise InputSerializationError("Execution input names must be strings.")
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise InputSerializationError(
                f"Execution input {name!r} is not JSON serializable."
            ) from None
        database_inputs.append({"name": name, "value": value})

    try:
        return json.dumps(
            database_inputs,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise InputSerializationError("Execution inputs are not JSON serializable.") from None


def execute(
    *,
    connection: oracledb.Connection,
    workflow: Executable,
    inputs: Mapping[str, JSONValue],
) -> JSONValue:
    """Execute a workflow using a caller-owned Oracle Database connection.

    Runs the supplied workflow in the database and returns its response as a
    Python value. Inputs are checked for JSON compatibility; validate external
    values against the workflow's input requirements in application code.

    The SDK closes its own cursor and leaves the connection open. It does not
    commit or roll back. Database-side tools can have their own side effects.
    Response reading, JSON decoding, and resource cleanup can fail after
    execution has completed; check side effects before retrying an operation.

    Loops and coordination between agents execute in the database and respect
    their configured limits. The SDK does not automatically retry failed work.

    :param connection: Open python-oracledb connection owned by the application.
        Connection setup, transaction control, and closure remain the caller's
        responsibility.
    :type connection: oracledb.Connection
    :param workflow: SDK-owned agent, node, or composed workflow. Create the
        workflow with the SDK's public functions.
    :type workflow: Executable
    :param inputs: Mapping of string input names to JSON-compatible values.
        Example: ``{"question": "How do I update my contact details?"}``.
        Supply the inputs declared by the workflow; non-finite input numbers
        such as NaN and infinity are rejected. A loop takes its body's named fields directly.
    :type inputs: Mapping[str, JSONValue]

    Returns:
        ``JSONValue`` containing the decoded database response: an object, array,
        string, number, boolean, or ``None`` for JSON null. Ordinary workflows
        return the decoded database response unchanged, including the final named
        state fields of a loop.

    Raises:
        BuildingBlockError: The workflow is not an SDK-owned executable.
        PatternCompilationError: A standalone node cannot be compiled as a flow.
        AgentSpecSerializationError: Workflow serialization fails.
        InputSerializationError: Inputs cannot be encoded as supported JSON.
        LoopExecutionError: A loop still requests continuation at its iteration limit.
        OrchestrationExecutionError: The manager selected an unknown capability.
        DatabaseExecutionError: Database execution or response reading fails.
        DatabaseCleanupError: Execution completes but cursor cleanup fails.
        InvalidDatabaseResponseError: The response is absent, blank, non-text,
            or invalid JSON."""
    component = _unwrap_executable(workflow)
    agent_spec_json = _serialize_workflow(workflow)
    inputs_json = _serialize_inputs(inputs)

    raw_result = execute_immediate(
        connection,
        agent_spec_json=agent_spec_json,
        inputs_json=inputs_json,
    )
    result = _decode_response(raw_result)
    field = status_field(component)
    if field is not None:
        completion = result.get(field) if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or not isinstance(completion, str)
            or completion
            not in {
                "complete",
                *failure_statuses(component),
            }
        ):
            raise InvalidDatabaseResponseError(
                "Database loop result has no valid completion status."
            )
        if result[field] == "exhausted":
            raise LoopExecutionError("Loop iteration limit reached before completion.")
        if result[field] == "orchestrator_invalid_action":
            raise OrchestrationExecutionError("Orchestrator selected an unknown capability.")
        return {key: value for key, value in result.items() if key != field}
    return result


def _decode_response(raw_result: object) -> JSONValue:
    """Decode the result of one completed database execution."""
    if raw_result is None or (isinstance(raw_result, str) and not raw_result.strip()):
        raise InvalidDatabaseResponseError("Database returned no agent response.")
    if not isinstance(raw_result, str):
        raise InvalidDatabaseResponseError("Database agent response must be JSON text.")

    try:
        return cast(
            JSONValue,
            json.loads(
                raw_result,
                parse_constant=_reject_nonstandard_json_constant,
                parse_float=_finite_float,
            ),
        )
    except (json.JSONDecodeError, ValueError, RecursionError):
        raise InvalidDatabaseResponseError("Database agent response is not valid JSON.") from None
