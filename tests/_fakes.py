##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""In-memory driver substitutes for testing binds, cleanup, and error handling.

These fakes implement only the operations exercised by the SDK tests. They do not
connect to a database or reproduce database transaction and SQL semantics.
"""

from __future__ import annotations

from typing import Any

import oracledb


class FakeVar:
    """Store bound values, rejecting text assignment to CLOBs like driver 26."""

    def __init__(self, database_type: Any = None, *, set_error: Exception | None = None) -> None:
        self.database_type = database_type
        self.set_error = set_error
        self.value: Any = None

    def setvalue(self, position: int, value: Any) -> None:
        assert position == 0
        if (
            self.database_type == oracledb.DB_TYPE_CLOB
            and value is not None
            and not isinstance(value, FakeLob)
        ):
            raise TypeError("CLOB setvalue requires a LOB object")
        if self.set_error is not None:
            raise self.set_error
        self.value = value

    def getvalue(self) -> Any:
        return self.value


class FakeLob:
    """Model temporary LOB writes and reads without allowing unpaired close()."""

    def __init__(
        self,
        value: Any = "",
        *,
        error: Exception | None = None,
        write_error: Exception | None = None,
    ) -> None:
        self.value = value
        self.error = error
        self.write_error = write_error
        self.read_called = False
        self.write_called = False
        self.close_called = False
        self.closed = False

    def read(self) -> Any:
        self.read_called = True
        assert not self.closed
        if self.error is not None:
            raise self.error
        return self.value

    def write(self, value: str) -> None:
        """Record a write, preserving allocation if writing fails."""
        self.write_called = True
        assert not self.closed
        if self.write_error is not None:
            raise self.write_error
        self.value = value

    def close(self) -> None:
        """Reject unpaired close(), which is not a temporary-LOB free operation."""
        self.close_called = True
        raise RuntimeError("LOB.close requires an earlier LOB.open")


class FakeCursor:
    """Record binds and calls, with configurable results and failure points."""

    def __init__(
        self,
        *,
        result: Any = None,
        error: Exception | None = None,
        close_error: Exception | None = None,
        var_error_on: int | None = None,
        var_error: Exception | None = None,
        bind_error_on: int | None = None,
        bind_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.close_error = close_error
        self.var_error_on = var_error_on
        self.var_error = var_error
        self.bind_error_on = bind_error_on
        self.bind_error = bind_error
        self.vars: list[tuple[Any, FakeVar]] = []
        self.executions: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def var(self, database_type: Any) -> FakeVar:
        """Record the requested database type and allocate an in-memory variable."""
        position = len(self.vars)
        if position == self.var_error_on:
            raise self.var_error or RuntimeError("Variable allocation failed")
        variable = FakeVar(
            database_type,
            set_error=self.bind_error if position == self.bind_error_on else None,
        )
        self.vars.append((database_type, variable))
        return variable

    def execute(self, statement: str, **bindings: Any) -> None:
        """Record a call and populate its result bind without executing SQL."""
        self.executions.append((statement, bindings))
        for name, variable in bindings.items():
            if name != "result" and isinstance(variable, FakeVar):
                assert isinstance(variable.value, FakeLob)
                assert not variable.value.closed
        if self.error is not None:
            raise self.error
        result = bindings.get("result")
        if result is not None:
            # Simulate a database OUT bind, not a user call to CLOB setvalue().
            # Result fixtures intentionally include invalid and plain-text values.
            result.value = self.result

    def close(self) -> None:
        """Record attempted cleanup, including when a cleanup error is configured."""
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeConnection:
    """Expose a fake cursor, temporary LOBs, and connection-lifecycle operations.

    The flags let tests verify that SDK operations do not take ownership of the
    caller's connection or transaction.
    """

    def __init__(
        self,
        cursor: FakeCursor,
        *,
        lobs: list[FakeLob] | None = None,
        cursor_error: Exception | None = None,
        lob_error_on: int | None = None,
        lob_error: Exception | None = None,
    ) -> None:
        self.fake_cursor = cursor
        self._configured_lobs = list(lobs or [])
        self.cursor_error = cursor_error
        self.lob_error_on = lob_error_on
        self.lob_error = lob_error
        self.lob_types: list[Any] = []
        self.created_lobs: list[FakeLob] = []
        self.cursor_called = False
        self.close_called = False
        self.commit_called = False
        self.rollback_called = False

    def cursor(self) -> FakeCursor:
        self.cursor_called = True
        if self.cursor_error is not None:
            raise self.cursor_error
        return self.fake_cursor

    def createlob(self, database_type: Any) -> FakeLob:
        """Allocate a configured temporary CLOB or fail at a selected position."""
        assert database_type == oracledb.DB_TYPE_CLOB
        position = len(self.lob_types)
        self.lob_types.append(database_type)
        if position == self.lob_error_on:
            raise self.lob_error or RuntimeError("Temporary CLOB allocation failed")
        lob = (
            self._configured_lobs[position] if position < len(self._configured_lobs) else FakeLob()
        )
        self.created_lobs.append(lob)
        return lob

    def close(self) -> None:
        self.close_called = True

    def commit(self) -> None:
        self.commit_called = True

    def rollback(self) -> None:
        self.rollback_called = True
