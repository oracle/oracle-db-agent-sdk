##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Execute database operations and manage SDK-owned resources."""

from __future__ import annotations

from typing import Any

import oracledb

from ._plsql import (
    CREATE_CREDENTIAL_PLSQL,
    EXECUTE_IMMEDIATE_PLSQL,
)
from .exceptions import (
    CredentialCreationError,
    DatabaseCleanupError,
    DatabaseExecutionError,
)


def _read_clob(value: Any) -> Any:
    """Read a LOB-like value, leaving text and other values for caller validation."""
    if isinstance(value, str) or value is None:
        return value
    read = getattr(value, "read", None)
    return read() if callable(read) else value


def _database_error_message(message: str, error: Exception) -> str:
    """Include the original database or driver diagnostic after SDK context."""
    diagnostic = str(error)
    return f"{message}\n{diagnostic}" if diagnostic else message


def _clob_bind(
    connection: oracledb.Connection,
    cursor: Any,
    value: str,
    owned_lobs: list[oracledb.LOB],
) -> Any:
    """Bind text through a temporary LOB, as required by python-oracledb 26.

    Register the LOB before writing or binding so either failure still triggers
    cleanup. Never assign strings directly to a CLOB variable's ``setvalue()``.
    """
    variable = cursor.var(oracledb.DB_TYPE_CLOB)
    lob = connection.createlob(oracledb.DB_TYPE_CLOB)
    owned_lobs.append(lob)
    lob.write(value)
    variable.setvalue(0, lob)
    return variable


def _close_sdk_resources(
    cursor: Any,
    owned_lobs: list[oracledb.LOB],
) -> Exception | None:
    """Close the cursor and release input LOB references, never the connection.

    The driver frees temporary LOBs when they and their bind variables lose all
    references. ``LOB.close()`` is not a free operation and requires an earlier
    ``LOB.open()``; do not call it for these single-write, implicitly opened LOBs.
    The caller also drops its bind-variable references, including on error.
    """
    first_error: Exception | None = None
    if cursor is not None:
        try:
            cursor.close()
        except Exception as exc:
            # Do not retain a failing driver's frame and its bound LOBs.
            first_error = exc.with_traceback(None)
    owned_lobs.clear()
    return first_error


def execute_immediate(
    connection: oracledb.Connection,
    *,
    agent_spec_json: str,
    inputs_json: str,
) -> Any:
    """Execute bound AgentSpec JSON and read the response before closing the cursor.

    Return the response without JSON decoding; the public executor validates its
    type and contents. Preserve connection ownership and transaction control.
    Execution or read failures take precedence over a simultaneous cleanup error.
    Input LOBs are owned by the SDK and are released along with its cursor.
    """
    cursor: Any = None
    operation_error: Exception | None = None
    owned_lobs: list[oracledb.LOB] = []
    database_result: Any = None
    database_call_completed = False
    try:
        cursor = connection.cursor()
        agent_spec_clob = _clob_bind(connection, cursor, agent_spec_json, owned_lobs)
        inputs_clob = _clob_bind(connection, cursor, inputs_json, owned_lobs)
        result_clob = cursor.var(oracledb.DB_TYPE_CLOB)
        cursor.execute(
            EXECUTE_IMMEDIATE_PLSQL,
            result=result_clob,
            agent_spec=agent_spec_clob,
            inputs=inputs_clob,
        )
        database_call_completed = True
        database_result = _read_clob(result_clob.getvalue())
    except Exception as exc:
        # Discard driver frames, which can retain temporary LOBs and input text.
        operation_error = exc.with_traceback(None)
    finally:
        cleanup_error = _close_sdk_resources(cursor, owned_lobs)
        cursor = agent_spec_clob = inputs_clob = result_clob = None

    if operation_error is not None:
        message = (
            "Database agent execution completed, but its response could not be read."
            if database_call_completed
            else "Database agent execution failed."
        )
        raise DatabaseExecutionError(_database_error_message(message, operation_error)) from None
    if cleanup_error is not None:
        raise DatabaseCleanupError(
            _database_error_message(
                "Database agent execution completed, but its cursor could not be closed.",
                cleanup_error,
            )
        ) from None
    return database_result


def create_database_credential(
    connection: oracledb.Connection,
    *,
    credential_name: str,
    credential_parameters_json: str,
) -> None:
    """Create a credential with bound values and close only SDK-owned resources.

    Leave connection ownership and transaction control to the caller. Report a
    creation failure before a simultaneous cursor-cleanup failure. Include the
    original database or driver diagnostic in the public exception message.
    """
    cursor: Any = None
    operation_error: Exception | None = None
    owned_lobs: list[oracledb.LOB] = []
    try:
        cursor = connection.cursor()
        parameters_clob = _clob_bind(connection, cursor, credential_parameters_json, owned_lobs)
        cursor.execute(
            CREATE_CREDENTIAL_PLSQL,
            credential_name=credential_name,
            params_json=parameters_clob,
        )
    except Exception as exc:
        operation_error = exc.with_traceback(None)
    finally:
        cleanup_error = _close_sdk_resources(cursor, owned_lobs)
        cursor = parameters_clob = None

    if operation_error is not None:
        raise CredentialCreationError(
            _database_error_message("Database credential creation failed.", operation_error)
        ) from None
    if cleanup_error is not None:
        raise DatabaseCleanupError(
            _database_error_message(
                "Database credential creation completed, but its cursor could not be closed.",
                cleanup_error,
            )
        ) from None
