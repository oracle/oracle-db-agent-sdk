##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""Demonstrate three independent claim-review branches in one workflow.

Before calling run_claim_reviews(), replace the illustrative LLM endpoint and
model settings and provision CLAIMS_LLM_CREDENTIAL. Use synthetic or appropriately
authorized input data. The caller supplies an open database connection; importing
this module does not execute the workflow. Runtime concurrency is database-managed.
"""

from __future__ import annotations

from collections.abc import Mapping

import oracledb

from oracle_db_agent_sdk import JSONValue, execute, llm_config, llm_node, parallel


def run_claim_reviews(
    connection: oracledb.Connection,
    inputs: Mapping[str, JSONValue],
) -> JSONValue:
    """Run medical, policy, and fraud review branches.

    Args:
        connection: Open, caller-owned connection with database agent access.
        inputs: String values for ``claim_id``, ``medical_notes``,
            ``policy_number``, and ``transaction_history``.

    Returns:
        The decoded database JSON response. The workflow declares distinct
        ``medical_result``, ``policy_result``, and ``fraud_result`` outputs.

    Raises:
        OracleDBAgentsError: SDK construction, serialization, or execution fails.

    This example illustrates composition, not validated medical, insurance, or
    fraud decision-making. It leaves connection and transaction control to the
    caller and does not guarantee a particular concurrency level or speedup.
    """
    review_llm = llm_config(
        provider="openai",
        model_id="gpt-5",
        url="https://LLM_URL",
        credential_name="CLAIMS_LLM_CREDENTIAL",
    )
    medical_review = llm_node(
        name="medical_review",
        llm=review_llm,
        prompt="Review the medical notes for claim {{claim_id}}: {{medical_notes}}",
        inputs={"claim_id": "string", "medical_notes": "string"},
        outputs={"medical_result": "string"},
    )
    policy_review = llm_node(
        name="policy_review",
        llm=review_llm,
        prompt="Review policy {{policy_number}} for claim {{claim_id}}.",
        inputs={"claim_id": "string", "policy_number": "string"},
        outputs={"policy_result": "string"},
    )
    fraud_review = llm_node(
        name="fraud_review",
        llm=review_llm,
        prompt="Review the transaction history: {{transaction_history}}",
        inputs={"transaction_history": "string"},
        outputs={"fraud_result": "string"},
    )
    workflow = parallel(name="claim_review", steps=[medical_review, policy_review, fraud_review])

    return execute(
        connection=connection,
        workflow=workflow,
        inputs=inputs,
    )
