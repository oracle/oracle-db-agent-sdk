##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Demonstrate ticket classification followed by one support destination.

Before calling run_support_ticket(), replace the illustrative LLM endpoint and
model settings and provision SUPPORT_LLM_CREDENTIAL. The caller supplies an open
database connection; importing this module does not execute the workflow.
"""

from __future__ import annotations

from collections.abc import Mapping

import oracledb

from oracle_db_agent_sdk import JSONValue, execute, llm_config, llm_node, routing


def run_support_ticket(
    connection: oracledb.Connection,
    inputs: Mapping[str, JSONValue],
) -> JSONValue:
    """Classify a ticket and execute only the selected support branch.

    Args:
        connection: Open, caller-owned connection with database agent access.
        inputs: String values for ``ticket_text`` and ``customer_id``.

    Returns:
        The decoded database JSON response. All destinations declare the same
        ``resolution`` output contract.

    Raises:
        OracleDBAgentsError: SDK construction, serialization, or execution fails.

    Destinations receive the original ticket text, not the classifier's output.
    Unmatched classifications use the general-support branch. Connection closure
    and transaction control remain the caller's responsibility.
    """
    support_llm = llm_config(
        provider="openai",
        model_id="gpt-5",
        url="https://LLM_URL",
        credential_name="SUPPORT_LLM_CREDENTIAL",
    )
    classify_ticket = llm_node(
        name="classify_ticket",
        llm=support_llm,
        prompt=(
            "Classify the ticket for customer {{customer_id}} as billing, security, "
            "or technical: {{ticket_text}}"
        ),
        inputs={"ticket_text": "string", "customer_id": "string"},
        outputs={"issue_type": "string"},
    )

    def build_support_branch(name: str, instruction: str):
        """Give each destination the same ticket input and resolution output."""
        return llm_node(
            name=name,
            llm=support_llm,
            prompt=f"{instruction}: {{{{ticket_text}}}}",
            inputs={"ticket_text": "string"},
            outputs={"resolution": "string"},
        )

    billing = build_support_branch("billing_support", "Resolve the billing request")
    security = build_support_branch("security_support", "Resolve the security request")
    technical = build_support_branch("technical_support", "Resolve the technical request")
    general = build_support_branch("general_support", "Resolve the general request")
    workflow = routing(
        classify_ticket,
        name="ticket_routing",
        route_key="issue_type",
        routes={
            "billing": billing,
            "security": security,
            "technical": technical,
        },
        default=general,
    )

    return execute(
        connection=connection,
        workflow=workflow,
        inputs=inputs,
    )
