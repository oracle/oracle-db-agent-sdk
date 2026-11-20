##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for the internal bound PL/SQL adapter."""

import gc
import json
import traceback
import weakref

import oracledb
import pytest

from oracle_db_agent_sdk import (
    CredentialCreationError,
    DatabaseCleanupError,
    DatabaseExecutionError,
)
from oracle_db_agent_sdk._database_adapter import (
    _database_error_message,
    create_database_credential,
    execute_immediate,
)

from ._fakes import FakeConnection, FakeCursor, FakeLob, FakeVar


@pytest.mark.parametrize("result", ['{"ok":true}', FakeLob('{"ok":true}')])
def test_execute_immediate_uses_clob_binds_and_caller_connection(result) -> None:
    cursor = FakeCursor(result=result)
    connection = FakeConnection(cursor)
    agent_json = '{"private":"agent-payload"}'
    inputs_json = '{"question":"input-payload"}'

    raw = execute_immediate(
        connection,  # type: ignore[arg-type]
        agent_spec_json=agent_json,
        inputs_json=inputs_json,
    )

    statement, bindings = cursor.executions[0]
    assert raw == '{"ok":true}'
    assert "DBMS_AI_AGENT.EXECUTE_IMMEDIATE" in statement
    assert "agent-payload" not in statement
    assert "input-payload" not in statement
    assert bindings["agent_spec"].value is connection.created_lobs[0]
    assert bindings["inputs"].value is connection.created_lobs[1]
    assert bindings["agent_spec"].value.value == agent_json
    assert bindings["inputs"].value.value == inputs_json
    assert connection.lob_types == [oracledb.DB_TYPE_CLOB, oracledb.DB_TYPE_CLOB]
    assert all(lob.write_called and not lob.close_called for lob in connection.created_lobs)
    assert [database_type for database_type, _ in cursor.vars] == [
        oracledb.DB_TYPE_CLOB,
        oracledb.DB_TYPE_CLOB,
        oracledb.DB_TYPE_CLOB,
    ]
    assert cursor.closed
    assert connection.cursor_called
    assert not connection.close_called
    assert not connection.commit_called
    assert not connection.rollback_called


def test_lob_is_read_before_cursor_is_closed() -> None:
    lob = FakeLob('{"ok":true}')
    cursor = FakeCursor(result=lob)
    connection = FakeConnection(cursor)

    execute_immediate(connection, agent_spec_json="{}", inputs_json="{}")  # type: ignore[arg-type]

    assert lob.read_called
    assert cursor.closed


def test_database_driver_error_preserves_diagnostic_without_adding_payload() -> None:
    original = RuntimeError("Driver execution diagnostic")
    cursor = FakeCursor(error=original)
    connection = FakeConnection(cursor)
    agent_spec_json = '{"secret":"synthetic-sensitive-value"}'

    with pytest.raises(DatabaseExecutionError) as caught:
        execute_immediate(  # type: ignore[arg-type]
            connection,
            agent_spec_json=agent_spec_json,
            inputs_json="{}",
        )

    assert caught.value.__cause__ is None
    assert str(caught.value) == "Database agent execution failed.\nDriver execution diagnostic"
    assert "synthetic-sensitive-value" not in str(caught.value)
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))
    assert cursor.closed
    assert not connection.close_called


def test_database_error_preserves_full_oracle_error_stack() -> None:
    class OracleErrorDetails:
        code = 49076

        def __str__(self) -> str:
            return "ORA-49076: invalid workflow field\nORA-06512: at line 4"

    cursor = FakeCursor(error=RuntimeError(OracleErrorDetails()))
    connection = FakeConnection(cursor)

    with pytest.raises(DatabaseExecutionError) as caught:
        execute_immediate(  # type: ignore[arg-type]
            connection,
            agent_spec_json="{}",
            inputs_json="{}",
        )

    assert str(caught.value) == (
        "Database agent execution failed.\nORA-49076: invalid workflow field\nORA-06512: at line 4"
    )


def test_response_read_error_preserves_diagnostic_and_operation_context() -> None:
    lob = FakeLob("unused", error=RuntimeError("synthetic-sensitive-value"))
    cursor = FakeCursor(result=lob)
    connection = FakeConnection(cursor)

    with pytest.raises(
        DatabaseExecutionError,
        match="execution completed, but its response could not be read",
    ) as caught:
        execute_immediate(  # type: ignore[arg-type]
            connection,
            agent_spec_json="{}",
            inputs_json="{}",
        )

    assert caught.value.__cause__ is None
    assert "synthetic-sensitive-value" in "".join(traceback.format_exception(caught.value))
    assert cursor.closed


def test_cursor_close_error_after_success_is_reported_as_cleanup_failure() -> None:
    cursor = FakeCursor(
        result='{"ok":true}',
        close_error=RuntimeError("synthetic-sensitive-value"),
    )
    connection = FakeConnection(cursor)

    with pytest.raises(DatabaseCleanupError, match="cursor could not be closed") as caught:
        execute_immediate(  # type: ignore[arg-type]
            connection,
            agent_spec_json="{}",
            inputs_json="{}",
        )

    assert caught.value.__cause__ is None
    assert "synthetic-sensitive-value" in "".join(traceback.format_exception(caught.value))


def test_execution_error_takes_priority_over_cursor_close_error() -> None:
    cursor = FakeCursor(
        error=RuntimeError("operation-sensitive-value"),
        close_error=RuntimeError("cleanup-sensitive-value"),
    )
    connection = FakeConnection(cursor)

    with pytest.raises(DatabaseExecutionError) as caught:
        execute_immediate(  # type: ignore[arg-type]
            connection,
            agent_spec_json="{}",
            inputs_json="{}",
        )

    rendered = "".join(traceback.format_exception(caught.value))
    assert "operation-sensitive-value" in rendered
    assert "cleanup-sensitive-value" not in rendered


@pytest.fixture(
    params=[
        pytest.param(
            (
                execute_immediate,
                {"agent_spec_json": "{}", "inputs_json": "{}"},
                DatabaseExecutionError,
                2,
            ),
            id="execution",
        ),
        pytest.param(
            (
                create_database_credential,
                {
                    "credential_name": "APP.TEST_CREDENTIAL",
                    "credential_parameters_json": '{"access_token":"synthetic-value"}',
                },
                CredentialCreationError,
                1,
            ),
            id="credential",
        ),
    ]
)
def clob_operation(request):
    """Exercise each adapter entry point against identical LOB failure scenarios."""
    function, arguments, error_type, count = request.param

    def invoke(connection):
        return function(connection, **arguments)

    return invoke, error_type, count


def _assert_connection_ownership(connection) -> None:
    assert not connection.close_called
    assert not connection.commit_called
    assert not connection.rollback_called


def _assert_sdk_cleanup_attempted(connection) -> None:
    assert connection.fake_cursor.closed
    assert all(not lob.close_called for lob in connection.created_lobs)
    _assert_connection_ownership(connection)


def _assert_diagnostic_preserved(error) -> None:
    assert error.__cause__ is None
    assert "private-value" in "".join(traceback.format_exception(error))


def test_disconnected_driver_error_retains_dpy_code(clob_operation) -> None:
    invoke, error_type, _ = clob_operation
    # Allocate the driver's disconnected state without connecting to a database.
    connection = oracledb.Connection.__new__(oracledb.Connection)

    with pytest.raises(error_type) as caught:
        invoke(connection)

    assert "DPY-1001: not connected to database" in str(caught.value)
    assert "ORA-00000" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("full_code", ["DPY-1001", "DPI-1010", "ORA-01031"])
@pytest.mark.parametrize("failure", ["operation", "cleanup"])
def test_adapter_errors_preserve_driver_diagnostics(clob_operation, full_code, failure) -> None:
    invoke, error_type, _ = clob_operation
    original = oracledb.DatabaseError(f"{full_code}: private-value")
    connection = FakeConnection(
        FakeCursor(
            result="{}",
            error=original if failure == "operation" else None,
            close_error=original if failure == "cleanup" else None,
        )
    )

    with pytest.raises(DatabaseCleanupError if failure == "cleanup" else error_type) as caught:
        invoke(connection)

    assert str(caught.value).endswith(f"\n{full_code}: private-value")
    assert "ORA-00000" not in str(caught.value)
    _assert_diagnostic_preserved(caught.value)
    _assert_sdk_cleanup_attempted(connection)


@pytest.mark.parametrize(
    "diagnostic",
    [
        "",
        "Driver diagnostic without an error code",
        "DPY-1001: not connected to database",
        "ORA-01031: insufficient privileges\nORA-06512: at line 2\n",
        "ORA-49076: invalid field 数据库 🧠",
    ],
)
def test_database_diagnostic_text_is_preserved_verbatim(diagnostic) -> None:
    error = RuntimeError(diagnostic)

    expected = f"Operation failed.\n{diagnostic}" if diagnostic else "Operation failed."
    assert _database_error_message("Operation failed.", error) == expected


@pytest.mark.parametrize("value", ["text", b"text", 123])
def test_clob_fake_rejects_non_lob_input_like_driver_26(value) -> None:
    variable = FakeVar(oracledb.DB_TYPE_CLOB)
    with pytest.raises(TypeError, match="CLOB setvalue requires a LOB object"):
        variable.setvalue(0, value)


def test_clob_fake_accepts_a_lob_and_null() -> None:
    variable = FakeVar(oracledb.DB_TYPE_CLOB)
    lob = FakeLob("text")
    variable.setvalue(0, lob)
    assert variable.getvalue() is lob
    variable.setvalue(0, None)
    assert variable.getvalue() is None


@pytest.mark.parametrize("text", ["x" * 70000, "数据库 🧠 café " * 10000])
def test_large_and_unicode_json_inputs_are_written_without_truncation(text) -> None:
    connection = FakeConnection(FakeCursor(result="{}"))
    payload = json.dumps({"text": text}, ensure_ascii=False)
    assert len(payload.encode("utf-8")) > 32767

    execute_immediate(  # type: ignore[arg-type]
        connection, agent_spec_json=payload, inputs_json=payload
    )

    assert [lob.value for lob in connection.created_lobs] == [payload, payload]
    assert all(not lob.close_called for lob in connection.created_lobs)
    _assert_sdk_cleanup_attempted(connection)


def test_all_adapter_operations_close_only_sdk_cursor_without_unpaired_lob_close(clob_operation):
    invoke, _, count = clob_operation
    connection = FakeConnection(FakeCursor(result="{}"))
    invoke(connection)
    assert len(connection.created_lobs) == count
    assert connection.lob_types == [oracledb.DB_TYPE_CLOB] * count
    assert all(lob.write_called and not lob.close_called for lob in connection.created_lobs)
    _assert_sdk_cleanup_attempted(connection)


def test_cursor_allocation_failure_creates_no_lobs(clob_operation) -> None:
    invoke, error_type, _ = clob_operation
    connection = FakeConnection(FakeCursor(), cursor_error=RuntimeError("private-value"))
    with pytest.raises(error_type) as caught:
        invoke(connection)
    assert connection.cursor_called
    assert not connection.fake_cursor.closed
    assert not connection.lob_types
    _assert_diagnostic_preserved(caught.value)
    _assert_connection_ownership(connection)


def test_variable_allocation_failures_release_already_created_lobs(clob_operation) -> None:
    invoke, error_type, count = clob_operation
    # Execution also allocates a CLOB OUT variable after creating both inputs.
    variable_count = count + (error_type is DatabaseExecutionError)
    for position in range(variable_count):
        cursor = FakeCursor(var_error_on=position, var_error=RuntimeError("private-value"))
        connection = FakeConnection(cursor)
        with pytest.raises(error_type) as caught:
            invoke(connection)
        assert len(connection.created_lobs) == min(position, count)
        assert not cursor.executions
        assert all(not lob.close_called for lob in connection.created_lobs)
        _assert_diagnostic_preserved(caught.value)
        _assert_sdk_cleanup_attempted(connection)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_lob_allocation_failure_releases_earlier_inputs(clob_operation, cleanup_fails) -> None:
    invoke, error_type, count = clob_operation
    for position in range(count):
        cursor = FakeCursor(close_error=RuntimeError("private-value") if cleanup_fails else None)
        connection = FakeConnection(
            cursor,
            lob_error_on=position,
            lob_error=RuntimeError("private-value"),
        )
        with pytest.raises(error_type) as caught:
            invoke(connection)
        assert len(connection.created_lobs) == position
        assert not cursor.executions
        _assert_diagnostic_preserved(caught.value)
        _assert_sdk_cleanup_attempted(connection)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_lob_write_failure_still_releases_the_new_lob(clob_operation, cleanup_fails) -> None:
    invoke, error_type, count = clob_operation
    for position in range(count):
        lobs = [FakeLob() for _ in range(count)]
        lobs[position].write_error = RuntimeError("private-value")
        connection = FakeConnection(
            FakeCursor(close_error=RuntimeError("private-value") if cleanup_fails else None),
            lobs=lobs,
        )
        with pytest.raises(error_type) as caught:
            invoke(connection)
        assert lobs[position].write_called
        assert len(connection.created_lobs) == position + 1
        assert not connection.fake_cursor.executions
        _assert_diagnostic_preserved(caught.value)
        _assert_sdk_cleanup_attempted(connection)


def test_lob_binding_failure_releases_every_allocated_lob(clob_operation) -> None:
    invoke, error_type, count = clob_operation
    for position in range(count):
        connection = FakeConnection(
            FakeCursor(bind_error_on=position, bind_error=RuntimeError("private-value"))
        )
        with pytest.raises(error_type) as caught:
            invoke(connection)
        assert len(connection.created_lobs) == position + 1
        assert all(lob.write_called and not lob.close_called for lob in connection.created_lobs)
        assert not connection.fake_cursor.executions
        _assert_diagnostic_preserved(caught.value)
        _assert_sdk_cleanup_attempted(connection)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_operation_error_precedes_cursor_cleanup_error(clob_operation, cleanup_fails) -> None:
    invoke, error_type, count = clob_operation
    connection = FakeConnection(
        FakeCursor(
            error=RuntimeError("private-value"),
            close_error=RuntimeError("private-value") if cleanup_fails else None,
        ),
    )
    with pytest.raises(error_type) as caught:
        invoke(connection)
    assert len(connection.created_lobs) == count
    _assert_diagnostic_preserved(caught.value)
    _assert_sdk_cleanup_attempted(connection)


def test_cursor_cleanup_failure_still_releases_all_lobs(clob_operation) -> None:
    invoke, _, count = clob_operation
    connection = FakeConnection(FakeCursor(result="{}", close_error=RuntimeError("private-value")))
    with pytest.raises(DatabaseCleanupError, match="cursor could not be closed") as caught:
        invoke(connection)
    assert len(connection.created_lobs) == count
    assert all(not lob.close_called for lob in connection.created_lobs)
    _assert_diagnostic_preserved(caught.value)
    _assert_sdk_cleanup_attempted(connection)


class _OracleErrorDetails:
    """Provide a database diagnostic for cleanup-error selection."""

    def __init__(self, code: int) -> None:
        self.code = code

    def __str__(self) -> str:
        return f"ORA-{abs(self.code):05d}: private-value"


def test_cursor_cleanup_error_preserves_oracle_diagnostic(clob_operation) -> None:
    invoke, _, count = clob_operation
    connection = FakeConnection(
        FakeCursor(result="{}", close_error=RuntimeError(_OracleErrorDetails(-1031))),
    )
    with pytest.raises(DatabaseCleanupError, match="cursor could not be closed") as caught:
        invoke(connection)
    assert str(caught.value).endswith("\nORA-01031: private-value")
    assert "ORA-03113" not in str(caught.value)
    assert len(connection.created_lobs) == count
    _assert_diagnostic_preserved(caught.value)
    _assert_sdk_cleanup_attempted(connection)


def test_output_read_failure_still_releases_input_lobs() -> None:
    output = FakeLob(error=RuntimeError("private-value"))
    connection = FakeConnection(FakeCursor(result=output))
    with pytest.raises(DatabaseExecutionError, match="response could not be read") as caught:
        execute_immediate(connection, agent_spec_json="{}", inputs_json="{}")  # type: ignore[arg-type]
    assert output.read_called
    assert all(not lob.close_called for lob in connection.created_lobs)
    _assert_diagnostic_preserved(caught.value)
    _assert_sdk_cleanup_attempted(connection)


class _ReleaseTrackingCursor(FakeCursor):
    """Drop recorder references at cursor closure, like releasing driver binds."""

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.vars.clear()
            self.executions.clear()


class _ReleaseTrackingConnection(FakeConnection):
    """Observe LOB lifetime without the usual fake's strong recorder references."""

    def __init__(self, cursor, *, write_error=False, **kwargs) -> None:
        super().__init__(cursor, **kwargs)
        self.lob_refs = []
        self.write_error = write_error

    def createlob(self, database_type):
        lob = super().createlob(database_type)
        if self.write_error:
            lob.write_error = RuntimeError("private-value")
        self.lob_refs.append(weakref.ref(lob))
        self.created_lobs.clear()
        return lob


@pytest.mark.parametrize("failure", [None, "allocation", "write", "bind", "execute", "cleanup"])
def test_temporary_lob_references_are_released_even_if_public_error_is_retained(
    clob_operation, failure
) -> None:
    invoke, error_type, count = clob_operation
    cursor = _ReleaseTrackingCursor(
        result="{}",
        error=RuntimeError("private-value") if failure == "execute" else None,
        close_error=RuntimeError("private-value") if failure == "cleanup" else None,
        bind_error_on=count - 1 if failure == "bind" else None,
        bind_error=RuntimeError("private-value") if failure == "bind" else None,
    )
    connection = _ReleaseTrackingConnection(
        cursor,
        write_error=failure == "write",
        lob_error_on=count - 1 if failure == "allocation" else None,
        lob_error=RuntimeError("private-value") if failure == "allocation" else None,
    )
    retained_error = None
    if failure is None:
        invoke(connection)
    else:
        expected_error = DatabaseCleanupError if failure == "cleanup" else error_type
        with pytest.raises(expected_error) as caught:
            invoke(connection)
        retained_error = caught.value
        _assert_diagnostic_preserved(retained_error)

    gc.collect()
    assert all(reference() is None for reference in connection.lob_refs)
    assert cursor.closed
    _assert_connection_ownership(connection)


def test_lob_close_is_not_temporary_lob_cleanup() -> None:
    lob = FakeLob()
    with pytest.raises(RuntimeError, match="LOB.close requires an earlier LOB.open"):
        lob.close()
