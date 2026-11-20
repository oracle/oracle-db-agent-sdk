##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Coordinate a case-status lookup and an embedded Python-defined specialist.

Before calling run_case_support(), replace the illustrative model and endpoint,
provision SUPPORT_LLM_CREDENTIAL, and provide APP_USER.CASE_SERVICE.GET_STATUS
with the declared interface. The specialist is included in the workflow. The caller
supplies the connection; importing this module does not execute the workflow.
"""

from __future__ import annotations

from collections.abc import Mapping

import oracledb

from oracle_db_agent_sdk import (
    JSONValue,
    agent,
    agent_tool,
    execute,
    llm_config,
    orchestrator,
    plsql_tool,
)


def run_case_support(connection: oracledb.Connection, inputs: Mapping[str, JSONValue]) -> JSONValue:
    """Resolve a case using a lookup tool and a specialist sub-agent.

    :param connection: Application-owned connection to the agent-enabled database.
    :type connection: oracledb.Connection
    :param inputs: Mapping containing the string ``case_id`` input.
    :type inputs: Mapping[str, JSONValue]

    Returns:
        ``JSONValue`` containing the database's response.

    Raises:
        OracleDBAgentsError: If configuration, serialization, or execution fails.
    """
    lookup = plsql_tool(
        name="lookup_case",
        description="Look up a support case's current status.",
        schema="APP_USER",
        package="CASE_SERVICE",
        procedure="GET_STATUS",
        inputs={"case_id": "string"},
        outputs={"status": "string"},
    )
    support_llm = llm_config(
        provider="openai",
        model_id="gpt-5.4",
        url="https://LLM_URL",
        credential_name="SUPPORT_LLM_CREDENTIAL",
    )
    specialist = agent(
        name="case_specialist",
        llm=support_llm,
        instruction=(
            "Assess support case {{case_id}} with status {{status}}. "
            "Recommend a response using only the supplied facts."
        ),
        inputs={"case_id": "string", "status": "string"},
        outputs={"assessment": "string"},
    )
    assessment = agent_tool(
        name="get_case_assessment",
        description="Ask a specialist to assess the support case and recommend a response.",
        agent=specialist,
    )
    workflow = orchestrator(
        name="case_coordinator",
        llm=support_llm,
        instruction=(
            "Resolve support case {{case_id}}. "
            "Use lookup_case for its current status. Pass that actual status and case_id "
            "to get_case_assessment for specialist advice. "
            "Combine the findings into a customer response."
        ),
        inputs={"case_id": "string"},
        outputs={"resolution": "string"},
        tools=[lookup, assessment],
    )
    return execute(connection=connection, workflow=workflow, inputs=inputs)
