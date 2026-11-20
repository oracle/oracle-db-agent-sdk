##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Demonstrate a PL/SQL lookup followed by an LLM explanation.

Before calling run_order_support(), replace the illustrative LLM endpoint and
model settings, provision SUPPORT_LLM_CREDENTIAL, and provide an accessible
ORDER_SERVICE.GET_STATUS tool procedure. The caller supplies an open database
connection; importing this module does not execute the workflow.
"""

from __future__ import annotations

from collections.abc import Mapping

import oracledb

from oracle_db_agent_sdk import (
    JSONValue,
    execute,
    llm_config,
    llm_node,
    plsql_tool,
    sequential,
    tool_node,
)


def run_order_support(
    connection: oracledb.Connection,
    inputs: Mapping[str, JSONValue],
) -> JSONValue:
    """Look up an order and explain its status to the customer.

    Args:
        connection: Open, caller-owned connection with agent and tool access.
        inputs: Workflow inputs containing ``order_id`` as a string.

    Returns:
        The decoded database JSON response. The workflow declares an ``answer``
        output; the SDK does not extract that field from the response.

    Raises:
        OracleDBAgentsError: SDK construction, serialization, or execution fails.

    Leaves connection closure and transaction control to the caller.
    """
    support_llm = llm_config(
        provider="openai",
        model_id="gpt-5",
        url="https://LLM_URL",
        credential_name="SUPPORT_LLM_CREDENTIAL",
    )
    lookup_order = tool_node(
        name="lookup_order",
        tool=plsql_tool(
            name="lookup_order_status",
            procedure="GET_STATUS",
            package="ORDER_SERVICE",
            inputs={"order_id": "string"},
            outputs={"order_status": "string"},
        ),
    )
    explain_status = llm_node(
        name="explain_status",
        llm=support_llm,
        prompt="Explain {{order_status}} clearly to the customer.",
        inputs={"order_status": "string"},
        outputs={"answer": "string"},
    )
    workflow = sequential(
        name="order_support",
        steps=[lookup_order, explain_status],
    )

    return execute(
        connection=connection,
        workflow=workflow,
        inputs=inputs,
    )
