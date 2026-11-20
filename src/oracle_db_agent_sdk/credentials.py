##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Public helpers for provisioning database-managed LLM credentials."""

from __future__ import annotations

import json

import oracledb

from ._database_adapter import create_database_credential
from .exceptions import CredentialConfigurationError


def create_api_key_credential(
    *,
    connection: oracledb.Connection,
    name: str,
    api_key: str,
) -> None:
    """Create, but never replace, a database-managed API-key credential.

    Stores the API key in a database-managed credential. Create credentials
    during application setup, then pass only their names to
    ``llm_config(credential_name=...)``. Keys are not included in workflow definitions.

    This helper provides no credential replacement or rotation operation. It
    closes its own cursor but does not commit, roll back, or close the connection.
    If cursor cleanup fails, creation may already have completed; verify the
    credential before retrying.

    :param connection: Open, caller-owned connection authorized to create
        database credentials.
    :type connection: oracledb.Connection
    :param name: Name for the database-managed credential. Must be a nonblank
        string. Example: ``"SUPPORT_LLM_CREDENTIAL"``.
    :type name: str
    :param api_key: Provider API key supplied through secure application
        configuration. Must be a nonblank string. Do not embed it in source,
        workflow definitions, or logs.
    :type api_key: str

    Returns:
        ``None``. The credential is created in the database; no credential object
        or secret value is returned to the caller.

    Raises:
        CredentialConfigurationError: The name or API key is not a nonblank string.
        CredentialCreationError: The database credential-creation call fails.
        DatabaseCleanupError: Creation completes but cursor cleanup fails."""
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(api_key, str)
        or not api_key.strip()
    ):
        raise CredentialConfigurationError("Credential name and API key must be non-empty strings.")

    credential_parameters_json = json.dumps(
        {"access_token": api_key},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    create_database_credential(
        connection,
        credential_name=name,
        credential_parameters_json=credential_parameters_json,
    )
