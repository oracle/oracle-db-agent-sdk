##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Tests for safe database credential provisioning."""

import json
import traceback

import oracledb
import pytest

import oracle_db_agent_sdk.credentials as credentials_module
from oracle_db_agent_sdk import (
    CredentialConfigurationError,
    CredentialCreationError,
    DatabaseCleanupError,
    create_api_key_credential,
)
from oracle_db_agent_sdk._database_adapter import create_database_credential

from ._fakes import FakeConnection, FakeCursor


def test_public_helper_builds_transient_access_token_document(monkeypatch) -> None:
    captured = {}

    def fake_create_database_credential(
        connection,
        *,
        credential_name,
        credential_parameters_json,
    ):
        captured.update(
            connection=connection,
            credential_name=credential_name,
            credential_parameters_json=credential_parameters_json,
        )

    monkeypatch.setattr(
        credentials_module,
        "create_database_credential",
        fake_create_database_credential,
    )
    connection = object()

    create_api_key_credential(
        connection=connection,  # type: ignore[arg-type]
        name="APP.TEST_CREDENTIAL",
        api_key="synthetic-sensitive-value",
    )

    assert captured["connection"] is connection
    assert captured["credential_name"] == "APP.TEST_CREDENTIAL"
    assert json.loads(captured["credential_parameters_json"]) == {
        "access_token": "synthetic-sensitive-value"
    }


def test_create_database_credential_uses_binds_without_managing_transaction() -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    credential_parameters_json = '{"access_token":"synthetic-sensitive-value"}'

    create_database_credential(  # type: ignore[arg-type]
        connection,
        credential_name="APP.TEST_CREDENTIAL",
        credential_parameters_json=credential_parameters_json,
    )

    statement, bindings = cursor.executions[0]
    assert "DBMS_VECTOR_CHAIN.CREATE_CREDENTIAL" in statement
    assert "APP.TEST_CREDENTIAL" not in statement
    assert "synthetic-sensitive-value" not in statement
    assert bindings["credential_name"] == "APP.TEST_CREDENTIAL"
    assert bindings["params_json"].value is connection.created_lobs[0]
    assert bindings["params_json"].value.value == credential_parameters_json
    assert connection.lob_types == [oracledb.DB_TYPE_CLOB]
    assert connection.created_lobs[0].write_called
    assert not connection.created_lobs[0].close_called
    assert cursor.closed
    assert not connection.close_called
    assert not connection.commit_called
    assert not connection.rollback_called


def test_credential_database_error_preserves_diagnostic_without_adding_secret() -> None:
    original = RuntimeError("ORA-01031: insufficient privileges")
    cursor = FakeCursor(error=original)
    connection = FakeConnection(cursor)
    credential_parameters_json = '{"access_token":"synthetic-sensitive-value"}'

    with pytest.raises(CredentialCreationError) as caught:
        create_database_credential(  # type: ignore[arg-type]
            connection,
            credential_name="APP.TEST_CREDENTIAL",
            credential_parameters_json=credential_parameters_json,
        )

    assert caught.value.__cause__ is None
    assert str(caught.value) == (
        "Database credential creation failed.\nORA-01031: insufficient privileges"
    )
    assert "synthetic-sensitive-value" not in str(caught.value)
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))
    assert cursor.closed


@pytest.mark.parametrize(
    ("name", "api_key"),
    [
        ("", "valid-key"),
        ("   ", "valid-key"),
        (None, "valid-key"),
        ("APP.TEST_CREDENTIAL", ""),
        ("APP.TEST_CREDENTIAL", "   "),
        ("APP.TEST_CREDENTIAL", object()),
    ],
)
def test_public_helper_rejects_invalid_configuration_before_database_call(
    name,
    api_key,
    monkeypatch,
) -> None:
    called = False

    def should_not_execute(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        credentials_module,
        "create_database_credential",
        should_not_execute,
    )

    with pytest.raises(
        CredentialConfigurationError,
        match="Credential name and API key must be non-empty strings",
    ):
        create_api_key_credential(
            connection=object(),  # type: ignore[arg-type]
            name=name,  # type: ignore[arg-type]
            api_key=api_key,  # type: ignore[arg-type]
        )

    assert not called


def test_credential_cursor_close_error_is_reported_as_cleanup_failure() -> None:
    cursor = FakeCursor(close_error=RuntimeError("Driver cursor cleanup diagnostic"))
    connection = FakeConnection(cursor)
    credential_parameters_json = '{"access_token":"synthetic-sensitive-value"}'

    with pytest.raises(DatabaseCleanupError, match="cursor could not be closed") as caught:
        create_database_credential(  # type: ignore[arg-type]
            connection,
            credential_name="APP.TEST_CREDENTIAL",
            credential_parameters_json=credential_parameters_json,
        )

    assert caught.value.__cause__ is None
    assert "Driver cursor cleanup diagnostic" in str(caught.value)
    assert "synthetic-sensitive-value" not in "".join(traceback.format_exception(caught.value))
