##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Keep parameter, return, and error documentation on public API functions."""

import inspect

import pytest

from oracle_db_agent_sdk import (
    agent,
    agent_tool,
    create_api_key_credential,
    execute,
    llm_config,
    llm_node,
    loop,
    orchestrator,
    parallel,
    plsql_tool,
    routing,
    sequential,
    tool_node,
)


@pytest.mark.parametrize(
    "factory",
    [
        agent,
        agent_tool,
        create_api_key_credential,
        execute,
        llm_config,
        llm_node,
        loop,
        orchestrator,
        parallel,
        plsql_tool,
        routing,
        sequential,
        tool_node,
    ],
)
def test_public_api_documents_every_parameter_and_result(factory) -> None:
    docstring = inspect.getdoc(factory)
    assert docstring
    for name in inspect.signature(factory).parameters:
        assert f":param {name}:" in docstring
        assert f":type {name}:" in docstring
    assert "Returns:" in docstring
    assert "Raises:" in docstring


def test_loop_docstring_explains_iteration_and_transaction_behavior() -> None:
    docstring = inspect.getdoc(loop)
    assert "database\npasses outputs to the next iteration as inputs" in docstring
    assert "max_iterations" in docstring
    assert "caller owns the connection and transaction" in docstring
    assert "DatabaseExecutionError" in docstring
